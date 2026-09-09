// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// clang-format on

#include <cuda_fp16.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/unique.h>
#include <thrust/universal_vector.h>

#include <fstream>
#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "cuembed/include/embedding_lookup.cuh"
#include "cuembed/include/index_transforms.cuh"
#include "utils/include/datagen.h"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"

// clang-format off
ABSL_FLAG(int, num_categories, 1048576,
          "Number of categories/rows of embedding");
ABSL_FLAG(int, embed_width, 128,
          "Width of embedding vector");
ABSL_FLAG(int, batch_size, 1024,
          "Batch size");
ABSL_FLAG(int, hotness, 1,
          "Number of nonzero indices per sample");
ABSL_FLAG(int, iterations, 1,
          "Number of iterations to run benchmark");
ABSL_FLAG(float, alpha, 0.,
          "alpha of power distribution. Use uniform if alpha is 0");
ABSL_FLAG(bool, use_int64_indices, false,
          "If true, use int64_t type for lookup indices.");
ABSL_FLAG(bool, check_result, false,
          "If true, compare GPU result against CPU reference");
ABSL_FLAG(bool, half_embedding_type, false,
          "If true, use fp16 for embedding");
ABSL_FLAG(bool, csr_input, false,
          "If true, use CSR formats for embedding lookup indices");
ABSL_FLAG(bool, weighted_sum, false,
          "If true, summation of rows would be weighted");
ABSL_FLAG(bool, fp16_math, false,
          "If true, fp16 embed rows will be use fp16 math during reduction."
          "This flag has no effect when the embed rows are in fp32.");
ABSL_FLAG(bool, compressed_grad, true,
          "If true, will compute a sparse gradient in the backward pass.");
ABSL_FLAG(bool, skip_grad_init, true,
          "If true, will skip the zero-initializion of the gradient "
          "during backward.");
ABSL_FLAG(bool, forward_only, false,
          "If true, will run only forward, skipping transpose and backward.");
ABSL_FLAG(bool, enable_csv, false,
          "If true, will output results in CSV format.");
ABSL_FLAG(bool, enable_stderr, true,
          "If true, will set stderr log level to INFO.");
ABSL_FLAG(bool, clear_caches, true,
          "If true, will clear caches between invocations by summing the "
          "full embedding table.");
ABSL_FLAG(bool, permute_indices, true,
          "If true, randomly permute generated category ids before lookup. "
          "This scatters power-law-hot logical ids across physical rows.");
ABSL_FLAG(bool, shuffle_indices, true,
          "If true, shuffle the order of generated ids within each sample.");
ABSL_FLAG(int64_t, l2_persist_start_row, 0,
          "Physical embedding row at which the optional L2 persistence "
          "region starts.");
ABSL_FLAG(int64_t, l2_persist_rows, 0,
          "Number of whole embedding rows in the optional L2 persistence "
          "region. Zero disables row-count configuration.");
ABSL_FLAG(int64_t, l2_persist_region_bytes, 0,
          "Size in bytes of the optional L2 persistence region. Must be a "
          "whole number of embedding rows. Zero disables byte-size "
          "configuration.");
#ifdef CUEMBED_CACHE_HINT_BENCHMARK
ABSL_FLAG(int64_t, l2_evict_last_rows, 0,
          "Number of physical prefix rows to load with L2::evict_last.");
ABSL_FLAG(int64_t, l2_evict_last_region_bytes, 0,
          "Size in bytes of the physical prefix to load with L2::evict_last. "
          "Must be a whole number of embedding rows.");
ABSL_FLAG(std::string, l2_secondary_hint, "evict_unchanged",
          "Secondary L2 hint: evict_unchanged or evict_first.");
ABSL_FLAG(bool, l2_use_range_policy, false,
          "Use createpolicy.range instead of row-cutoff load selection.");
#endif
// clang-format on

struct L2PersistenceConfig {
  bool enabled{false};
  int64_t requested_start_row{0};
  int64_t requested_rows{0};
  int64_t requested_bytes{0};
  int64_t effective_start_row{0};
  int64_t effective_rows{0};
  int64_t effective_bytes{0};
  int64_t reserved_bytes{0};
  float hit_ratio{0.0F};
  size_t previous_set_aside_bytes{0};
};

struct CacheHintBenchmarkConfig {
  cuembed::CacheEvictionHintConfig forward_config{};
  int64_t requested_rows{0};
  int64_t requested_bytes{0};
  int64_t effective_rows{0};
  int64_t effective_bytes{0};
};

template <typename ElemT>
CacheHintBenchmarkConfig ConfigureCacheHints(const int num_categories,
                                             const int embed_width,
                                             const int64_t requested_rows,
                                             const int64_t requested_bytes,
                                             const std::string& secondary_hint,
                                             const bool use_range_policy) {
  CacheHintBenchmarkConfig config;
  config.requested_rows = requested_rows;
  config.requested_bytes = requested_bytes;
  if (requested_rows == 0 && requested_bytes == 0) return config;
  if (requested_rows < 0 || requested_bytes < 0 ||
      (requested_rows > 0 && requested_bytes > 0)) {
    LOG(FATAL) << "Specify exactly one non-negative cache-priority row or byte value.";
  }
  const int64_t row_bytes = static_cast<int64_t>(embed_width) * sizeof(ElemT);
  int64_t rows = requested_rows;
  if (requested_bytes > 0) {
    if (requested_bytes % row_bytes != 0) {
      LOG(FATAL) << "--l2_evict_last_region_bytes must be a whole number of embedding rows.";
    }
    rows = requested_bytes / row_bytes;
  }
  if (rows <= 0 || rows > num_categories) {
    LOG(FATAL) << "Cache-priority prefix must be within the embedding table.";
  }
  const int64_t bytes = rows * row_bytes;
  if (bytes > UINT32_MAX) {
    LOG(FATAL) << "Cache-priority prefix exceeds PTX's 32-bit range-policy size limit.";
  }
  if (secondary_hint == "evict_first") {
    config.forward_config.secondary_hint = cuembed::L2SecondaryHint::kFirst;
  } else if (secondary_hint != "evict_unchanged") {
    LOG(FATAL) << "--l2_secondary_hint must be evict_unchanged or evict_first.";
  }
  config.forward_config.evict_last_rows = rows;
  config.forward_config.table_bytes =
      static_cast<size_t>(num_categories) * static_cast<size_t>(row_bytes);
  config.forward_config.use_range_policy = use_range_policy;
  if (use_range_policy && config.forward_config.table_bytes > UINT32_MAX) {
    LOG(FATAL) << "Range policy requires an embedding table no larger than 4 GB.";
  }
  config.effective_rows = rows;
  config.effective_bytes = bytes;
  return config;
}

template <typename ElemT>
L2PersistenceConfig ConfigureL2Persistence(
    ElemT* embedding,
    const int num_categories,
    const int embed_width,
    const int64_t start_row,
    const int64_t persist_rows,
    const int64_t persist_region_bytes) {
  L2PersistenceConfig config;
  if (persist_rows == 0 && persist_region_bytes == 0) {
    return config;
  }

  if (start_row < 0 || persist_rows < 0 || persist_region_bytes < 0) {
    LOG(FATAL) << "L2 persistence start row, row count, and byte size must "
               << "be non-negative.";
  }
  if (persist_rows > 0 && persist_region_bytes > 0) {
    LOG(FATAL) << "Specify exactly one of --l2_persist_rows or "
               << "--l2_persist_region_bytes.";
  }
  if (start_row >= num_categories) {
    LOG(FATAL) << "--l2_persist_start_row=" << start_row
               << " is outside the embedding table with " << num_categories
               << " rows.";
  }

  const int64_t row_bytes =
      static_cast<int64_t>(embed_width) * static_cast<int64_t>(sizeof(ElemT));
  CHECK_GT(row_bytes, 0);
  int64_t requested_rows = persist_rows;
  if (persist_region_bytes > 0) {
    if (persist_region_bytes % row_bytes != 0) {
      LOG(FATAL) << "--l2_persist_region_bytes=" << persist_region_bytes
                 << " must be a multiple of the embedding row size "
                 << row_bytes << " bytes.";
    }
    requested_rows = persist_region_bytes / row_bytes;
  }
  CHECK_GT(requested_rows, 0);
  if (requested_rows > num_categories - start_row) {
    LOG(FATAL) << "Requested L2 persistence rows [" << start_row << ", "
               << (start_row + requested_rows)
               << ") exceed the embedding table with " << num_categories
               << " rows.";
  }

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp device_properties{};
  CHECK_CUDA(cudaGetDeviceProperties(&device_properties, device));
  if (device_properties.major < 8 ||
      device_properties.accessPolicyMaxWindowSize == 0 ||
      device_properties.persistingL2CacheMaxSize == 0) {
    LOG(FATAL) << "L2 persistence is unavailable on device " << device
               << " (compute capability " << device_properties.major << "."
               << device_properties.minor
               << ", access-policy window max "
               << device_properties.accessPolicyMaxWindowSize
               << ", persisting L2 max "
               << device_properties.persistingL2CacheMaxSize
               << "). This can occur on unsupported devices or with MIG.";
  }

  const int64_t max_window_rows =
      static_cast<int64_t>(device_properties.accessPolicyMaxWindowSize) /
      row_bytes;
  if (max_window_rows == 0) {
    LOG(FATAL) << "Embedding row size " << row_bytes
               << " exceeds CUDA's access-policy window maximum "
               << device_properties.accessPolicyMaxWindowSize << " bytes.";
  }
  const int64_t effective_rows = std::min(requested_rows, max_window_rows);
  const int64_t effective_bytes = effective_rows * row_bytes;
  const size_t requested_set_aside = std::min(
      static_cast<size_t>(effective_bytes),
      static_cast<size_t>(device_properties.persistingL2CacheMaxSize));

  CHECK_CUDA(cudaDeviceGetLimit(&config.previous_set_aside_bytes,
                                cudaLimitPersistingL2CacheSize));
  CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                requested_set_aside));
  size_t actual_set_aside = 0;
  CHECK_CUDA(cudaDeviceGetLimit(&actual_set_aside,
                                cudaLimitPersistingL2CacheSize));
  if (actual_set_aside != requested_set_aside) {
    // CUDA may round or retain a larger device-level set-aside (for example,
    // when another runtime component has already configured it). The stream
    // policy remains valid as long as its hit ratio is in [0, 1], so record
    // the effective value and continue rather than rejecting the benchmark.
    LOG(WARNING) << "CUDA did not apply the requested persisting-L2 set-aside "
                 << requested_set_aside << " bytes exactly (actual "
                 << actual_set_aside << "); continuing with the effective "
                 << "set-aside.";
  }

  cudaStreamAttrValue stream_attribute{};
  stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(
      embedding + static_cast<size_t>(start_row) * embed_width);
  stream_attribute.accessPolicyWindow.num_bytes =
      static_cast<size_t>(effective_bytes);
  stream_attribute.accessPolicyWindow.hitRatio = std::min(
      1.0F, static_cast<float>(actual_set_aside) /
                 static_cast<float>(effective_bytes));
  stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  CHECK_CUDA(cudaStreamSetAttribute(
      0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));

  config.enabled = true;
  config.requested_start_row = start_row;
  config.requested_rows = requested_rows;
  config.requested_bytes = requested_rows * row_bytes;
  config.effective_start_row = start_row;
  config.effective_rows = effective_rows;
  config.effective_bytes = effective_bytes;
  config.reserved_bytes = static_cast<int64_t>(actual_set_aside);
  config.hit_ratio = stream_attribute.accessPolicyWindow.hitRatio;
  return config;
}

void ResetL2Persistence(const L2PersistenceConfig& config) {
  if (!config.enabled) {
    return;
  }
  cudaStreamAttrValue stream_attribute{};
  stream_attribute.accessPolicyWindow.base_ptr = nullptr;
  stream_attribute.accessPolicyWindow.num_bytes = 0;
  stream_attribute.accessPolicyWindow.hitRatio = 1.0F;
  stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  CHECK_CUDA(cudaStreamSetAttribute(
      0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));
  CHECK_CUDA(cudaCtxResetPersistingL2Cache());
  CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                config.previous_set_aside_bytes));
}

template <typename T>
void ValidateResult(const thrust::universal_vector<T>& result,
                    const thrust::universal_vector<T>& h_result) {
  CHECK_EQ(thrust::equal(result.begin(), result.end(), h_result.begin()), true);
  CHECK_EQ(result.size(), h_result.size());
}

std::string combine_mode_str(cuembed::CombineMode mode) {
  switch (mode) {
    case cuembed::CombineMode::kSum:
      return "kSum";
    case cuembed::CombineMode::kMean:
      return "kMean";
    case cuembed::CombineMode::kConcat:
      return "kConcat";
  }
  return "Unknown";
}

void dump_csv_header(std::ofstream& outfile) {
  outfile << "num_categories,batch_size,hotness,alpha,embed_width,combine_mode,"
             "permute_indices,shuffle_indices,is_csr,is_weighted,"
             "compressed_grad,skip_grad_init,name,iterations,elapsed_time_ms,"
             "avg_time_ms,algo_bw_l2,algo_bw_dram,l2_persist_enabled,"
             "l2_persist_requested_start_row,l2_persist_requested_rows,"
             "l2_persist_requested_bytes,l2_persist_effective_start_row,"
             "l2_persist_effective_rows,l2_persist_effective_bytes,"
             "l2_persist_reserved_bytes,l2_persist_hit_ratio"
#ifdef CUEMBED_CACHE_HINT_BENCHMARK
             ",l2_cache_hint_enabled,l2_cache_hint_range_policy,"
             "l2_cache_hint_secondary,l2_cache_hint_requested_rows,"
             "l2_cache_hint_requested_bytes,l2_cache_hint_effective_rows,"
             "l2_cache_hint_effective_bytes"
#endif
          << std::endl;
}

void dump_csv_line(std::ofstream& outfile,
                   const cuembed::utils::AllocationOptions& options,
                   std::string name,
                   int iterations,
                   double elapsed_time_ms,
                   double algo_bw_l2,
                   double algo_bw_dram,
                   const L2PersistenceConfig& l2_persistence
#ifdef CUEMBED_CACHE_HINT_BENCHMARK
                   , const CacheHintBenchmarkConfig& cache_hint
#endif
                   ) {
  outfile << options.num_categories() << "," << options.batch_size() << ","
          << options.hotness() << "," << options.alpha() << ","
          << options.embed_width() << ","
          << combine_mode_str(options.combine_mode()) << ","
          << options.permute_indices() << "," << options.shuffle_indices() << ","
          << options.is_csr() << "," << options.is_weighted() << ","
          << options.compressed_grad() << "," << options.skip_grad_init() << ","
          << name << "," << absl::StrFormat("%d ", iterations) << ","
          << absl::StrFormat("%.2f ", elapsed_time_ms) << ","
          << absl::StrFormat("%.6f ", elapsed_time_ms / iterations) << ","
          << absl::StrFormat("%.2f", algo_bw_l2) << ","
          << absl::StrFormat("%.2f", algo_bw_dram) << ","
          << l2_persistence.enabled << ","
          << l2_persistence.requested_start_row << ","
          << l2_persistence.requested_rows << ","
          << l2_persistence.requested_bytes << ","
          << l2_persistence.effective_start_row << ","
          << l2_persistence.effective_rows << ","
          << l2_persistence.effective_bytes << ","
          << l2_persistence.reserved_bytes << ","
          << l2_persistence.hit_ratio
#ifdef CUEMBED_CACHE_HINT_BENCHMARK
          << "," << (cache_hint.effective_rows > 0) << ","
          << cache_hint.forward_config.use_range_policy << ","
          << (cache_hint.forward_config.secondary_hint == cuembed::L2SecondaryHint::kFirst
                  ? "evict_first" : "evict_unchanged") << ","
          << cache_hint.requested_rows << "," << cache_hint.requested_bytes
          << "," << cache_hint.effective_rows << ","
          << cache_hint.effective_bytes
#endif
          << std::endl;
}

bool file_exists(const std::string& fname) {
  std::ifstream infile(fname);
  return infile.good();
}

template <typename ElemT>
void clear_cache(ElemT* clear_cache_max,
                 const thrust::device_vector<int>& clear_cache_buffer) {
  *clear_cache_max += thrust::reduce(thrust::device,
                                     clear_cache_buffer.begin(),
                                     clear_cache_buffer.end(),
                                     0,
                                     thrust::maximum<ElemT>());
}

namespace cuembed {
template <typename ElemT, typename IndexT, typename OffsetT, bool fp16_math>
void EmbeddingLookupBenchmark(const int num_categories,
                              const int embed_width,
                              const int batch_size,
                              const int hotness,
                              const float alpha,
                              const bool is_csr,
                              const bool is_weighted,
                              const bool compressed_grad,
                              const bool skip_grad_init,
                              const bool forward_only,
                              const bool check_result,
                              const int iterations,
                              const bool enable_csv,
                              const bool clear_caches,
                              const bool permute_indices,
                              const bool shuffle_indices,
                              const int64_t l2_persist_start_row,
                              const int64_t l2_persist_rows,
                              const int64_t l2_persist_region_bytes) {
  utils::AllocationOptions options;
  options.num_categories(num_categories)
      .batch_size(batch_size)
      .hotness(hotness)
      .alpha(alpha)
      .embed_width(embed_width)
      .combine_mode(CombineMode::kSum)
      .is_csr(is_csr)
      .is_weighted(is_weighted)
      .compressed_grad(compressed_grad)
      .skip_grad_init(skip_grad_init)
      .permute_indices(permute_indices)
      .shuffle_indices(shuffle_indices);

  std::ofstream outfile;
  if (enable_csv) {
    std::string fname =
#ifdef CUEMBED_CACHE_HINT_BENCHMARK
        "cache_hint_benchmark_out.csv";
#else
        "manual_benchmark_out.csv";
#endif
    bool existed_before = file_exists(fname);
    outfile.open(fname, std::ios::out | std::ios::app);

    if (!outfile.is_open()) {
      std::cerr << "Unable to open file\n";
      return;
    }

    if (!existed_before) {
      dump_csv_header(outfile);
    }
  }

  // Allocate buffers
  utils::
      UniversalEmbeddingAllocation<ElemT, IndexT, OffsetT, ElemT, ElemT, ElemT>
          u_a;
  utils::DeviceEmbeddingAllocation<ElemT, IndexT, OffsetT, ElemT, ElemT, ElemT>
      d_a;
  utils::AllocateHost(options, &u_a, forward_only);
  utils::AllocateDevice(options, u_a, &d_a, forward_only);
  const L2PersistenceConfig l2_persistence = ConfigureL2Persistence(
      d_a.embedding.data().get(),
      num_categories,
      embed_width,
      l2_persist_start_row,
      l2_persist_rows,
      l2_persist_region_bytes);
  CacheHintBenchmarkConfig cache_hint;
#ifdef CUEMBED_CACHE_HINT_BENCHMARK
  cache_hint = ConfigureCacheHints<ElemT>(
      num_categories,
      embed_width,
      absl::GetFlag(FLAGS_l2_evict_last_rows),
      absl::GetFlag(FLAGS_l2_evict_last_region_bytes),
      absl::GetFlag(FLAGS_l2_secondary_hint),
      absl::GetFlag(FLAGS_l2_use_range_policy));
#endif

  // Used for clearing caches
  thrust::device_vector<int> clear_cache_buffer;
  if (clear_caches) {
    clear_cache_buffer.resize(256000000L, 1);
  }
  ElemT clear_cache_max = static_cast<ElemT>(0);

  auto run_forward = [&]() {
#ifdef CUEMBED_CACHE_HINT_BENCHMARK
    utils::RunForward<ElemT, IndexT, OffsetT, fp16_math>(options,
                                                         d_a.embedding,
                                                         d_a.indices,
                                                         d_a.offsets,
                                                         d_a.weights,
                                                         &d_a.result,
                                                         cache_hint.forward_config);
#else
    utils::RunForward<ElemT, IndexT, OffsetT, fp16_math>(options,
                                                         d_a.embedding,
                                                         d_a.indices,
                                                         d_a.offsets,
                                                         d_a.weights,
                                                         &d_a.result);
#endif
  };

  // Warm up
  run_forward();

  if (clear_caches) {
    clear_cache(&clear_cache_max, clear_cache_buffer);
  }

  // Actual run and recording elapsed time.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed_time_ms = 0.0;

  for (int iter = 0; iter < iterations; iter++) {
    if (clear_caches || (iter == 0)) {
      cudaEventRecord(start);
    }

    run_forward();

    if (clear_caches || (iter == iterations - 1)) {
      cudaEventRecord(stop);
      CHECK_CUDA(cudaEventSynchronize(stop));

      float iter_elapsed_time_ms = 0.0;
      cudaEventElapsedTime(&iter_elapsed_time_ms, start, stop);
      elapsed_time_ms += iter_elapsed_time_ms;
    }

    if (clear_caches) {
      clear_cache(&clear_cache_max, clear_cache_buffer);
    }
  }

  double algo_bw = 0.0;
  if (options.is_csr()) {
    algo_bw = sizeof(ElemT) * iterations *
              (d_a.offsets.back() - 1 + options.batch_size()) *
              options.embed_width() / 1.e6 / elapsed_time_ms;
  } else {
    algo_bw = sizeof(ElemT) * iterations * options.batch_size() *
              (options.hotness() + (options.combine_mode() == CombineMode::kSum
                                        ? 1
                                        : options.hotness())) *
              options.embed_width() / 1.e6 / elapsed_time_ms;
  }

  if (enable_csv) {
    dump_csv_line(outfile,
                  options,
                  "forward",
                  iterations,
                  elapsed_time_ms,
                  algo_bw,
                  0.0 /*algo_bw_dram*/,
                  l2_persistence
#ifdef CUEMBED_CACHE_HINT_BENCHMARK
                  , cache_hint
#endif
                  );
  }
  LOG(INFO) << "Embedding forward. Iterations: "
            << absl::StrFormat("%d ", iterations) << ", Total time [ms]: "
            << absl::StrFormat("%.2f ", elapsed_time_ms) << ", Avg [ms]: "
            << absl::StrFormat("%.2f ", elapsed_time_ms / iterations)
            << ", Application BW [GB/s]: " << absl::StrFormat("%.2f", algo_bw);

  if (check_result) {
    utils::RunForwardReference<ElemT, IndexT, OffsetT, fp16_math>(options,
                                                                  u_a.embedding,
                                                                  u_a.indices,
                                                                  u_a.offsets,
                                                                  u_a.weights,
                                                                  &u_a.result);
    ValidateResult<ElemT>(d_a.result, u_a.result);
    LOG(INFO) << "Check result forward passed";
  }

  if (forward_only) {
    ResetL2Persistence(l2_persistence);
    return;
  }

  OffsetT nnz = static_cast<OffsetT>(d_a.indices.size());
  utils::RunTranspose<IndexT, OffsetT, ElemT>(options,
                                              d_a.indices,
                                              d_a.offsets,
                                              d_a.weights,
                                              nnz,
                                              &d_a.transpose_indices,
                                              &d_a.transpose_remapped_indices,
                                              &d_a.transpose_sample_ids,
                                              &d_a.transpose_weights,
                                              &d_a.sample_ids,
                                              &d_a.transpose_workspace);

  if (clear_caches) {
    clear_cache(&clear_cache_max, clear_cache_buffer);
  }

  float elapsed_time_ms_transpose = 0.0;
  for (int iter = 0; iter < iterations; iter++) {
    if (clear_caches || (iter == 0)) {
      cudaEventRecord(start);
    }

    utils::RunTranspose<IndexT, OffsetT, ElemT>(options,
                                                d_a.indices,
                                                d_a.offsets,
                                                d_a.weights,
                                                nnz,
                                                &d_a.transpose_indices,
                                                &d_a.transpose_remapped_indices,
                                                &d_a.transpose_sample_ids,
                                                &d_a.transpose_weights,
                                                &d_a.sample_ids,
                                                &d_a.transpose_workspace);
    if (clear_caches || (iter == iterations - 1)) {
      cudaEventRecord(stop);
      CHECK_CUDA(cudaEventSynchronize(stop));
      float iter_elapsed_time_ms = 0.0;
      cudaEventElapsedTime(&iter_elapsed_time_ms, start, stop);
      elapsed_time_ms_transpose += iter_elapsed_time_ms;
    }

    if (clear_caches) {
      clear_cache(&clear_cache_max, clear_cache_buffer);
    }
  }

  double algo_bw_transpose = 0.0;

  // Input
  algo_bw_transpose += nnz * sizeof(IndexT);
  algo_bw_transpose += (options.is_csr()) ? nnz * sizeof(OffsetT) : 0;
  algo_bw_transpose += (options.is_weighted()) ? nnz * sizeof(ElemT) : 0;

  // Output
  algo_bw_transpose +=
      ((options.compressed_grad()) ? 3 : 2) * nnz * sizeof(IndexT);
  algo_bw_transpose += (options.is_weighted()) ? nnz * sizeof(ElemT) : 0;

  algo_bw_transpose *= iterations;
  algo_bw_transpose /= (1.e6);
  algo_bw_transpose /= elapsed_time_ms_transpose;
  if (enable_csv) {
    dump_csv_line(outfile,
                  options,
                  "transpose",
                  iterations,
                  elapsed_time_ms_transpose,
                  0.0,
                  algo_bw_transpose,
                  l2_persistence
#ifdef CUEMBED_CACHE_HINT_BENCHMARK
                  , cache_hint
#endif
                  );
  }

  LOG(INFO) << "Transpose. Iterations: " << absl::StrFormat("%d ", iterations)
            << ", Total time [ms]: "
            << absl::StrFormat("%.2f ", elapsed_time_ms_transpose)
            << ", Avg [ms]: "
            << absl::StrFormat("%.2f ", elapsed_time_ms_transpose / iterations)
            << ", Application BW [GB/s]: "
            << absl::StrFormat("%.2f", algo_bw_transpose);

  if (check_result) {
    utils::RunTransposeReference<IndexT, OffsetT, ElemT>(
        options,
        u_a.indices,
        u_a.offsets,
        u_a.weights,
        nnz,
        &u_a.transpose_indices,
        &u_a.transpose_remapped_indices,
        &u_a.transpose_sample_ids,
        &u_a.transpose_weights);
    ValidateResult<IndexT>(d_a.transpose_indices, u_a.transpose_indices);
    if (options.compressed_grad()) {
      ValidateResult<IndexT>(d_a.transpose_remapped_indices,
                             u_a.transpose_remapped_indices);
    }
    LOG(INFO) << "Check results transpose passed";
  }

  int num_unique = (options.compressed_grad())
                       ? d_a.transpose_remapped_indices.back() + 1
                       : 0;

  utils::RunBackward<ElemT, IndexT, OffsetT>(options,
                                             d_a.grad_y,
                                             d_a.transpose_indices,
                                             d_a.transpose_remapped_indices,
                                             d_a.transpose_sample_ids,
                                             d_a.transpose_weights,
                                             d_a.offsets,
                                             nnz,
                                             num_unique,
                                             &d_a.grad_embedding,
                                             &d_a.inverse_mapping);

  if (clear_caches) {
    clear_cache(&clear_cache_max, clear_cache_buffer);
  }

  float elapsed_time_ms_backward = 0.0;
  for (int iter = 0; iter < iterations; iter++) {
    if (clear_caches || (iter == 0)) {
      cudaEventRecord(start);
    }

    utils::RunBackward<ElemT, IndexT, OffsetT>(options,
                                               d_a.grad_y,
                                               d_a.transpose_indices,
                                               d_a.transpose_remapped_indices,
                                               d_a.transpose_sample_ids,
                                               d_a.transpose_weights,
                                               d_a.offsets,
                                               nnz,
                                               num_unique,
                                               &d_a.grad_embedding,
                                               &d_a.inverse_mapping);

    if (clear_caches || (iter == iterations - 1)) {
      cudaEventRecord(stop);
      CHECK_CUDA(cudaEventSynchronize(stop));

      float iter_elapsed_time_ms = 0.0;
      cudaEventElapsedTime(&iter_elapsed_time_ms, start, stop);
      elapsed_time_ms_backward += iter_elapsed_time_ms;
    }

    if (clear_caches) {
      clear_cache(&clear_cache_max, clear_cache_buffer);
    }
  }

  double algo_bw_backward_dram = 0.0;

  // Need to compute number of unique indices for bandwidth calcs
  int num_unique_ = thrust::unique_count(thrust::device,
                                         d_a.transpose_indices.begin(),
                                         d_a.transpose_indices.begin() + nnz);

  // Writes to embedding weight gradient
  algo_bw_backward_dram += sizeof(ElemT) * options.embed_width() * num_unique_;

  // Reads of COO lookup indices
  algo_bw_backward_dram += sizeof(IndexT) * nnz * 2;
  algo_bw_backward_dram += (options.is_weighted()) ? sizeof(ElemT) * nnz : 0;

  // Reads from grad_y
  double algo_bw_backward_l2 = 0.0;
  if (options.combine_mode() == CombineMode::kConcat) {
    algo_bw_backward_dram += sizeof(ElemT) * options.embed_width() * nnz;
    algo_bw_backward_l2 = algo_bw_backward_dram;
  } else {
    algo_bw_backward_dram +=
        sizeof(ElemT) * options.embed_width() * options.batch_size();
    algo_bw_backward_l2 =
        algo_bw_backward_dram + sizeof(ElemT) * options.embed_width() * nnz;
  }

  algo_bw_backward_dram =
      (algo_bw_backward_dram * iterations) / 1e6 / elapsed_time_ms_backward;
  algo_bw_backward_l2 =
      (algo_bw_backward_l2 * iterations) / 1e6 / elapsed_time_ms_backward;

  if (enable_csv) {
    dump_csv_line(outfile,
                  options,
                  "backward",
                  iterations,
                  elapsed_time_ms_backward,
                  algo_bw_backward_l2,
                  algo_bw_backward_dram,
                  l2_persistence
#ifdef CUEMBED_CACHE_HINT_BENCHMARK
                  , cache_hint
#endif
                  );
  }

  LOG(INFO) << "Backward. Iterations: " << absl::StrFormat("%d ", iterations)
            << ", Total time [ms]: "
            << absl::StrFormat("%.2f ", elapsed_time_ms_backward)
            << ", Avg [ms]: "
            << absl::StrFormat("%.2f ", elapsed_time_ms_backward / iterations)
            << ", Application DRAM BW [GB/s]: "
            << absl::StrFormat("%.2f", algo_bw_backward_dram)
            << ", Application L2 BW [GB/s]: "
            << absl::StrFormat("%.2f", algo_bw_backward_l2);

  if (check_result) {
    utils::RunBackwardReference<ElemT, IndexT, OffsetT>(
        options,
        u_a.grad_y,
        u_a.transpose_indices,
        u_a.transpose_remapped_indices,
        u_a.transpose_sample_ids,
        u_a.transpose_weights,
        u_a.offsets,
        nnz,
        &u_a.grad_embedding,
        &u_a.inverse_mapping);
    ValidateResult<ElemT>(d_a.grad_embedding, u_a.grad_embedding);
    if (options.compressed_grad()) {
      ValidateResult<IndexT>(d_a.inverse_mapping, u_a.inverse_mapping);
    }
    LOG(INFO) << "Check result backward passed";
  }
  if (enable_csv) {
    outfile.close();
  }
  ResetL2Persistence(l2_persistence);
}

}  // namespace cuembed

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  int num_categories = absl::GetFlag(FLAGS_num_categories);
  int embed_width = absl::GetFlag(FLAGS_embed_width);
  int batch_size = absl::GetFlag(FLAGS_batch_size);
  int hotness = absl::GetFlag(FLAGS_hotness);
  int iterations = absl::GetFlag(FLAGS_iterations);
  float alpha = absl::GetFlag(FLAGS_alpha);
  bool check_result = absl::GetFlag(FLAGS_check_result);
  bool half_embedding_type = absl::GetFlag(FLAGS_half_embedding_type);
  bool use_int64_indices = absl::GetFlag(FLAGS_use_int64_indices);
  bool is_csr = absl::GetFlag(FLAGS_csr_input);
  bool is_weighted = absl::GetFlag(FLAGS_weighted_sum);
  bool fp16_math = absl::GetFlag(FLAGS_fp16_math);
  bool compressed_grad = absl::GetFlag(FLAGS_compressed_grad);
  bool skip_grad_init = absl::GetFlag(FLAGS_skip_grad_init);
  bool forward_only = absl::GetFlag(FLAGS_forward_only);
  bool enable_csv = absl::GetFlag(FLAGS_enable_csv);
  bool enable_stderr = absl::GetFlag(FLAGS_enable_stderr);
  bool clear_caches = absl::GetFlag(FLAGS_clear_caches);
  bool permute_indices = absl::GetFlag(FLAGS_permute_indices);
  bool shuffle_indices = absl::GetFlag(FLAGS_shuffle_indices);
  int64_t l2_persist_start_row = absl::GetFlag(FLAGS_l2_persist_start_row);
  int64_t l2_persist_rows = absl::GetFlag(FLAGS_l2_persist_rows);
  int64_t l2_persist_region_bytes =
      absl::GetFlag(FLAGS_l2_persist_region_bytes);
  LOG(INFO) << "parsed flag num_categories: " << num_categories
            << ", embed_width: " << embed_width
            << ", batch_size: " << batch_size << ", hotness: " << hotness
            << ", alpha: " << alpha
            << ", fp16 embedding: " << half_embedding_type
            << ", int64_t indices: " << use_int64_indices
            << ", csr indices: " << is_csr << ", weighted sum: " << is_weighted
            << ", fp16 math: " << fp16_math
            << ", sparse gradient: " << compressed_grad
            << ", skip gradient init: " << skip_grad_init
            << ", forward_only: " << forward_only
            << ", enable_csv: " << enable_csv
            << ", enable_stderr: " << enable_stderr
            << ", clear_caches: " << clear_caches
            << ", permute_indices: " << permute_indices
            << ", shuffle_indices: " << shuffle_indices
            << ", l2_persist_start_row: " << l2_persist_start_row
            << ", l2_persist_rows: " << l2_persist_rows
            << ", l2_persist_region_bytes: " << l2_persist_region_bytes;

  if (enable_stderr) {
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  } else {
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kFatal);
  }

  if (half_embedding_type && use_int64_indices && fp16_math) {
    cuembed::EmbeddingLookupBenchmark<__half, int64_t, int, true>(
        num_categories,
        embed_width,
        batch_size,
        hotness,
        alpha,
        is_csr,
        is_weighted,
        compressed_grad,
        skip_grad_init,
        forward_only,
        check_result,
        iterations,
        enable_csv,
        clear_caches,
        permute_indices,
        shuffle_indices,
        l2_persist_start_row,
        l2_persist_rows,
        l2_persist_region_bytes);
  } else if (half_embedding_type && !use_int64_indices && fp16_math) {
    cuembed::EmbeddingLookupBenchmark<__half, int32_t, int, true>(
        num_categories,
        embed_width,
        batch_size,
        hotness,
        alpha,
        is_csr,
        is_weighted,
        compressed_grad,
        skip_grad_init,
        forward_only,
        check_result,
        iterations,
        enable_csv,
        clear_caches,
        permute_indices,
        shuffle_indices,
        l2_persist_start_row,
        l2_persist_rows,
        l2_persist_region_bytes);
  } else if (half_embedding_type && use_int64_indices && !fp16_math) {
    cuembed::EmbeddingLookupBenchmark<__half, int64_t, int, false>(
        num_categories,
        embed_width,
        batch_size,
        hotness,
        alpha,
        is_csr,
        is_weighted,
        compressed_grad,
        skip_grad_init,
        forward_only,
        check_result,
        iterations,
        enable_csv,
        clear_caches,
        permute_indices,
        shuffle_indices,
        l2_persist_start_row,
        l2_persist_rows,
        l2_persist_region_bytes);
  } else if (half_embedding_type && !use_int64_indices && !fp16_math) {
    cuembed::EmbeddingLookupBenchmark<__half, int32_t, int, false>(
        num_categories,
        embed_width,
        batch_size,
        hotness,
        alpha,
        is_csr,
        is_weighted,
        compressed_grad,
        skip_grad_init,
        forward_only,
        check_result,
        iterations,
        enable_csv,
        clear_caches,
        permute_indices,
        shuffle_indices,
        l2_persist_start_row,
        l2_persist_rows,
        l2_persist_region_bytes);
  } else if (!half_embedding_type && use_int64_indices) {
    cuembed::EmbeddingLookupBenchmark<float, int64_t, int, true>(
        num_categories,
        embed_width,
        batch_size,
        hotness,
        alpha,
        is_csr,
        is_weighted,
        compressed_grad,
        skip_grad_init,
        forward_only,
        check_result,
        iterations,
        enable_csv,
        clear_caches,
        permute_indices,
        shuffle_indices,
        l2_persist_start_row,
        l2_persist_rows,
        l2_persist_region_bytes);
  } else if (!half_embedding_type && !use_int64_indices) {
    cuembed::EmbeddingLookupBenchmark<float, int32_t, int, true>(
        num_categories,
        embed_width,
        batch_size,
        hotness,
        alpha,
        is_csr,
        is_weighted,
        compressed_grad,
        skip_grad_init,
        forward_only,
        check_result,
        iterations,
        enable_csv,
        clear_caches,
        permute_indices,
        shuffle_indices,
        l2_persist_start_row,
        l2_persist_rows,
        l2_persist_region_bytes);
  }

  return 0;
}
