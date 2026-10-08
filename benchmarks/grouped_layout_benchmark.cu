// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "cuembed/include/embedding_lookup.cuh"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"
#include "utils/include/grouped_datagen.h"

ABSL_FLAG(int, num_categories, 1048576, "Number of embedding rows.");
ABSL_FLAG(int, embed_width, 128, "Embedding vector width.");
ABSL_FLAG(int, batch_size, 1024, "Queries per lookup invocation.");
ABSL_FLAG(int, hotness, 16, "Rows in every exclusive coaccess group.");
ABSL_FLAG(int, groups_per_bank, 16,
          "Equiprobable coaccess groups in each frequency bank.");
ABSL_FLAG(int, iterations, 1, "Number of timed replay invocations.");
ABSL_FLAG(double, alpha, 0.0,
          "Power-law exponent for frequency-bank selection; 0 is uniform.");
ABSL_FLAG(uint64_t, seed, 123456, "Base seed for independent RNG streams.");
ABSL_FLAG(std::string, frequency_layout, "sorted", "sorted or random.");
ABSL_FLAG(std::string, coaccess_layout, "compact", "compact or scattered.");
ABSL_FLAG(std::string, scatter_policy, "interleaved",
          "interleaved or random; effective only for scattered layouts.");
ABSL_FLAG(bool, shuffle_within_query, true,
          "Deterministically shuffle the members of each generated query.");
ABSL_FLAG(bool, use_int64_indices, true, "Use int64 rather than int32 indices.");
ABSL_FLAG(bool, half_embedding_type, false, "Use fp16 rather than fp32 rows.");
ABSL_FLAG(bool, fp16_math, false,
          "Accumulate fp16 embedding rows using fp16 arithmetic.");
ABSL_FLAG(bool, check_result, false,
          "Check the final GPU output against logical row values.");
ABSL_FLAG(bool, enable_csv, true, "Append the timing result to CSV output.");
ABSL_FLAG(std::string, csv_output, "grouped_layout_benchmark_out.csv",
          "CSV output path.");
ABSL_FLAG(bool, clear_caches, true,
          "Evict cache contents with a large reduction between invocations.");
ABSL_FLAG(int64_t, l2_persist_start_row, 0,
          "First physical row in the optional L2 persistence window.");
ABSL_FLAG(int64_t, l2_persist_rows, 0,
          "Rows in the optional L2 persistence window.");
ABSL_FLAG(int64_t, l2_persist_region_bytes, 0,
          "Bytes in the optional L2 persistence window; exclusive with rows.");

namespace {

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

cuembed::utils::FrequencyLayout ParseFrequencyLayout(
    const std::string& value) {
  if (value == "sorted") return cuembed::utils::FrequencyLayout::kSorted;
  if (value == "random") return cuembed::utils::FrequencyLayout::kRandom;
  throw std::invalid_argument("--frequency_layout must be sorted or random");
}

cuembed::utils::CoaccessLayout ParseCoaccessLayout(
    const std::string& value) {
  if (value == "compact") return cuembed::utils::CoaccessLayout::kCompact;
  if (value == "scattered") {
    return cuembed::utils::CoaccessLayout::kScattered;
  }
  throw std::invalid_argument("--coaccess_layout must be compact or scattered");
}

cuembed::utils::ScatterPolicy ParseScatterPolicy(const std::string& value) {
  if (value == "interleaved") {
    return cuembed::utils::ScatterPolicy::kInterleaved;
  }
  if (value == "random") return cuembed::utils::ScatterPolicy::kRandom;
  throw std::invalid_argument("--scatter_policy must be interleaved or random");
}

template <typename ElemT>
L2PersistenceConfig ConfigureL2Persistence(ElemT* embedding,
                                            int64_t num_categories,
                                            int embed_width,
                                            int64_t start_row,
                                            int64_t persist_rows,
                                            int64_t persist_bytes) {
  L2PersistenceConfig config;
  if (persist_rows == 0 && persist_bytes == 0) return config;
  if (start_row < 0 || persist_rows < 0 || persist_bytes < 0 ||
      (persist_rows > 0 && persist_bytes > 0)) {
    LOG(FATAL) << "Invalid L2 persistence row/byte configuration.";
  }
  const int64_t row_bytes = static_cast<int64_t>(embed_width) * sizeof(ElemT);
  int64_t rows = persist_rows;
  if (persist_bytes > 0) {
    if (persist_bytes % row_bytes != 0) {
      LOG(FATAL) << "--l2_persist_region_bytes must contain whole rows.";
    }
    rows = persist_bytes / row_bytes;
  }
  if (rows <= 0 || start_row >= num_categories ||
      rows > num_categories - start_row) {
    LOG(FATAL) << "L2 persistence range is outside the embedding table.";
  }

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
  if (properties.major < 8 || properties.accessPolicyMaxWindowSize == 0 ||
      properties.persistingL2CacheMaxSize == 0) {
    LOG(FATAL) << "L2 persistence is unavailable on this device.";
  }
  const int64_t effective_rows = std::min<int64_t>(
      rows, static_cast<int64_t>(properties.accessPolicyMaxWindowSize) /
                row_bytes);
  if (effective_rows <= 0) {
    LOG(FATAL) << "Embedding rows exceed the L2 policy-window maximum.";
  }
  const int64_t effective_bytes = effective_rows * row_bytes;
  const size_t requested_set_aside = std::min<size_t>(
      effective_bytes, properties.persistingL2CacheMaxSize);
  CHECK_CUDA(cudaDeviceGetLimit(&config.previous_set_aside_bytes,
                                cudaLimitPersistingL2CacheSize));
  CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                requested_set_aside));
  size_t actual_set_aside = 0;
  CHECK_CUDA(cudaDeviceGetLimit(&actual_set_aside,
                                cudaLimitPersistingL2CacheSize));

  cudaStreamAttrValue attribute{};
  attribute.accessPolicyWindow.base_ptr =
      embedding + static_cast<size_t>(start_row) * embed_width;
  attribute.accessPolicyWindow.num_bytes = effective_bytes;
  attribute.accessPolicyWindow.hitRatio = std::min(
      1.0F, static_cast<float>(actual_set_aside) / effective_bytes);
  attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  CHECK_CUDA(cudaStreamSetAttribute(
      0, cudaStreamAttributeAccessPolicyWindow, &attribute));

  config.enabled = true;
  config.requested_start_row = start_row;
  config.requested_rows = rows;
  config.requested_bytes = rows * row_bytes;
  config.effective_start_row = start_row;
  config.effective_rows = effective_rows;
  config.effective_bytes = effective_bytes;
  config.reserved_bytes = actual_set_aside;
  config.hit_ratio = attribute.accessPolicyWindow.hitRatio;
  return config;
}

void ResetL2Persistence(const L2PersistenceConfig& config) {
  if (!config.enabled) return;
  cudaStreamAttrValue attribute{};
  attribute.accessPolicyWindow.base_ptr = nullptr;
  attribute.accessPolicyWindow.num_bytes = 0;
  attribute.accessPolicyWindow.hitRatio = 1.0F;
  attribute.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  CHECK_CUDA(cudaStreamSetAttribute(
      0, cudaStreamAttributeAccessPolicyWindow, &attribute));
  CHECK_CUDA(cudaCtxResetPersistingL2Cache());
  CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                config.previous_set_aside_bytes));
}

__host__ __device__ float LogicalEmbeddingValue(int64_t logical_row,
                                                 int column) {
  const uint64_t mixed = static_cast<uint64_t>(logical_row) * 131ULL +
                         static_cast<uint64_t>(column) * 17ULL + 19ULL;
  return static_cast<float>(mixed % 1024ULL) / 1024.0F;
}

template <typename ElemT>
__global__ void InitializeEmbeddingKernel(ElemT* embedding,
                                          int embed_width,
                                          int64_t physical_start,
                                          int64_t row_count,
                                          const int64_t* physical_to_logical) {
  const int64_t element =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = row_count * embed_width;
  if (element >= total) return;
  const int64_t local_row = element / embed_width;
  const int column = static_cast<int>(element % embed_width);
  embedding[(physical_start + local_row) * embed_width + column] =
      static_cast<ElemT>(LogicalEmbeddingValue(
          physical_to_logical[local_row], column));
}

template <typename ElemT>
void InitializeEmbedding(
    thrust::device_vector<ElemT>* embedding,
    int64_t num_categories,
    int embed_width,
    const std::vector<int64_t>& physical_to_logical) {
  constexpr int64_t kChunkRows = 1 << 20;
  thrust::device_vector<int64_t> device_mapping;
  for (int64_t start = 0; start < num_categories; start += kChunkRows) {
    const int64_t rows = std::min(kChunkRows, num_categories - start);
    device_mapping.assign(physical_to_logical.begin() + start,
                          physical_to_logical.begin() + start + rows);
    const int64_t elements = rows * embed_width;
    const int blocks = static_cast<int>((elements + 255) / 256);
    InitializeEmbeddingKernel<<<blocks, 256>>>(
        thrust::raw_pointer_cast(embedding->data()), embed_width, start, rows,
        thrust::raw_pointer_cast(device_mapping.data()));
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename ElemT>
float ToFloat(ElemT value) {
  return static_cast<float>(value);
}

template <>
float ToFloat(__half value) {
  return __half2float(value);
}

template <typename ElemT, bool fp16_math>
void CheckResult(const thrust::device_vector<ElemT>& result,
                 const std::vector<int64_t>& logical_indices,
                 int batch_size,
                 int hotness,
                 int embed_width) {
  const thrust::host_vector<ElemT> host_result = result;
  const float tolerance = std::is_same_v<ElemT, __half>
                              ? (fp16_math ? 0.01F * hotness : 1e-3F)
                              : 1e-4F;
  for (int sample = 0; sample < batch_size; ++sample) {
    for (int column = 0; column < embed_width; ++column) {
      float expected = 0.0F;
      for (int hot = 0; hot < hotness; ++hot) {
        const int64_t logical =
            logical_indices[static_cast<size_t>(sample) * hotness + hot];
        expected += ToFloat(static_cast<ElemT>(
            LogicalEmbeddingValue(logical, column)));
      }
      const float actual = ToFloat(
          host_result[static_cast<size_t>(sample) * embed_width + column]);
      if (std::abs(actual - expected) > tolerance) {
        LOG(FATAL) << "Result mismatch at sample " << sample << ", column "
                   << column << ": actual=" << actual
                   << " expected=" << expected
                   << " tolerance=" << tolerance;
      }
    }
  }
}

bool FileExists(const std::string& path) {
  std::ifstream input(path);
  return input.good();
}

void DumpCsvHeader(std::ofstream& output) {
  output
      << "input_source,frequency_layout,coaccess_layout,scatter_policy,seed,"
         "trace_checksum,mapping_checksum,contiguous_groups,"
         "adjacent_member_pair_ratio,mean_group_span_rows,num_categories,num_frequency_banks,"
         "num_groups,groups_per_bank,batch_size,hotness,alpha,embed_width,"
         "half_embedding_type,use_int64_indices,fp16_math,shuffle_within_query,"
         "clear_caches,iterations,elapsed_time_ms,avg_time_ms,algo_bw_l2,"
         "l2_persist_enabled,l2_persist_requested_start_row,"
         "l2_persist_requested_rows,l2_persist_requested_bytes,"
         "l2_persist_effective_start_row,l2_persist_effective_rows,"
         "l2_persist_effective_bytes,l2_persist_reserved_bytes,"
         "l2_persist_hit_ratio\n";
}

template <typename ElemT, typename IndexT, bool fp16_math>
void RunBenchmark(const cuembed::utils::GroupedWorkloadConfig& config,
                  cuembed::utils::FrequencyLayout frequency_layout,
                  cuembed::utils::CoaccessLayout coaccess_layout,
                  cuembed::utils::ScatterPolicy scatter_policy) {
  if (config.num_categories > std::numeric_limits<int>::max()) {
    LOG(FATAL) << "cuEmbed AllocationOptions requires int32 num_categories.";
  }
  if (config.num_categories - 1 >
      static_cast<int64_t>(std::numeric_limits<IndexT>::max())) {
    LOG(FATAL) << "The configured index type cannot represent all rows.";
  }
  const int embed_width = absl::GetFlag(FLAGS_embed_width);
  const int iterations = absl::GetFlag(FLAGS_iterations);
  if (embed_width <= 0 || iterations <= 0) {
    LOG(FATAL) << "embed_width and iterations must be positive.";
  }

  const auto workload = cuembed::utils::GenerateGroupedWorkload(config);
  const auto mapping = cuembed::utils::BuildGroupedLayoutMapping(
      config, frequency_layout, coaccess_layout, scatter_policy);
  std::vector<IndexT> host_indices;
  host_indices.reserve(workload.logical_indices.size());
  for (const int64_t logical : workload.logical_indices) {
    host_indices.push_back(
        static_cast<IndexT>(mapping.logical_to_physical[logical]));
  }

  cuembed::utils::AllocationOptions options;
  options.num_categories(static_cast<int>(config.num_categories))
      .batch_size(config.batch_size)
      .hotness(config.hotness)
      .alpha(static_cast<float>(config.alpha))
      .embed_width(embed_width)
      .combine_mode(cuembed::CombineMode::kSum)
      .is_csr(false)
      .is_weighted(false);

  thrust::device_vector<ElemT> embedding(
      static_cast<size_t>(config.num_categories) * embed_width);
  thrust::device_vector<IndexT> indices(host_indices.begin(),
                                        host_indices.end());
  thrust::device_vector<int> offsets;
  thrust::device_vector<ElemT> weights;
  thrust::device_vector<ElemT> result(
      static_cast<size_t>(config.batch_size) * embed_width);
  InitializeEmbedding(&embedding, config.num_categories, embed_width,
                      mapping.physical_to_logical);

  const auto persistence = ConfigureL2Persistence(
      thrust::raw_pointer_cast(embedding.data()), config.num_categories,
      embed_width, absl::GetFlag(FLAGS_l2_persist_start_row),
      absl::GetFlag(FLAGS_l2_persist_rows),
      absl::GetFlag(FLAGS_l2_persist_region_bytes));

  auto run_forward = [&]() {
    cuembed::utils::RunForward<ElemT, IndexT, int, fp16_math>(
        options, embedding, indices, offsets, weights, &result);
  };

  thrust::device_vector<int> clear_buffer;
  volatile int clear_sink = 0;
  const bool clear_caches = absl::GetFlag(FLAGS_clear_caches);
  if (clear_caches) clear_buffer.assign(256000000L, 1);

  run_forward();
  CHECK_CUDA(cudaDeviceSynchronize());
  if (clear_caches) {
    clear_sink = thrust::reduce(clear_buffer.begin(), clear_buffer.end(), 0);
  }

  cudaEvent_t start_event, stop_event;
  CHECK_CUDA(cudaEventCreate(&start_event));
  CHECK_CUDA(cudaEventCreate(&stop_event));
  float elapsed_ms = 0.0F;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (clear_caches || iteration == 0) {
      CHECK_CUDA(cudaEventRecord(start_event));
    }
    run_forward();
    if (clear_caches || iteration + 1 == iterations) {
      CHECK_CUDA(cudaEventRecord(stop_event));
      CHECK_CUDA(cudaEventSynchronize(stop_event));
      float interval_ms = 0.0F;
      CHECK_CUDA(cudaEventElapsedTime(&interval_ms, start_event, stop_event));
      elapsed_ms += interval_ms;
    }
    if (clear_caches) {
      clear_sink = thrust::reduce(clear_buffer.begin(), clear_buffer.end(), 0);
    }
  }
  if (clear_sink == std::numeric_limits<int>::min()) std::cerr << clear_sink;

  if (absl::GetFlag(FLAGS_check_result)) {
    CheckResult<ElemT, fp16_math>(result, workload.logical_indices,
                                  config.batch_size, config.hotness,
                                  embed_width);
  }

  const double bytes = static_cast<double>(sizeof(ElemT)) * iterations *
                       config.batch_size * (config.hotness + 1) * embed_width;
  const double algo_bw = bytes / (elapsed_ms * 1.0e6);
  const std::string frequency_name =
      frequency_layout == cuembed::utils::FrequencyLayout::kSorted ? "sorted"
                                                                   : "random";
  const std::string coaccess_name =
      coaccess_layout == cuembed::utils::CoaccessLayout::kCompact
          ? "compact"
          : "scattered";
  const std::string effective_scatter =
      coaccess_layout == cuembed::utils::CoaccessLayout::kCompact
          ? "none"
          : (scatter_policy == cuembed::utils::ScatterPolicy::kInterleaved
                 ? "interleaved"
                 : "random");

  if (absl::GetFlag(FLAGS_enable_csv)) {
    const std::string path = absl::GetFlag(FLAGS_csv_output);
    const bool exists = FileExists(path);
    std::ofstream output(path, std::ios::app);
    if (!output) LOG(FATAL) << "Unable to open CSV output: " << path;
    if (!exists) DumpCsvHeader(output);
    output << std::setprecision(10) << "exclusive_groups," << frequency_name
           << "," << coaccess_name << "," << effective_scatter << ","
           << config.seed << ","
           << cuembed::utils::ChecksumHex(workload.trace_checksum) << ","
           << cuembed::utils::ChecksumHex(mapping.mapping_checksum) << ","
           << mapping.contiguous_groups << ","
           << mapping.adjacent_member_pair_ratio << ","
           << mapping.mean_group_span_rows << ","
           << config.num_categories << "," << workload.num_frequency_banks
           << "," << workload.num_groups << "," << config.groups_per_bank
           << "," << config.batch_size << "," << config.hotness << ","
           << config.alpha << "," << embed_width << ","
           << std::is_same_v<ElemT, __half> << ","
           << std::is_same_v<IndexT, int64_t> << "," << fp16_math << ","
           << config.shuffle_within_query << "," << clear_caches << ","
           << iterations << "," << elapsed_ms << ","
           << elapsed_ms / iterations << "," << algo_bw << ","
           << persistence.enabled << "," << persistence.requested_start_row
           << "," << persistence.requested_rows << ","
           << persistence.requested_bytes << ","
           << persistence.effective_start_row << ","
           << persistence.effective_rows << ","
           << persistence.effective_bytes << ","
           << persistence.reserved_bytes << "," << persistence.hit_ratio
           << "\n";
  }

  LOG(INFO) << "Grouped forward: frequency=" << frequency_name
            << " coaccess=" << coaccess_name
            << " scatter=" << effective_scatter
            << " trace="
            << cuembed::utils::ChecksumHex(workload.trace_checksum)
            << " mapping="
            << cuembed::utils::ChecksumHex(mapping.mapping_checksum)
            << " avg_ms=" << elapsed_ms / iterations
            << " application_bw_gbps=" << algo_bw;
  CHECK_CUDA(cudaEventDestroy(start_event));
  CHECK_CUDA(cudaEventDestroy(stop_event));
  ResetL2Persistence(persistence);
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  cuembed::utils::GroupedWorkloadConfig config;
  config.num_categories = absl::GetFlag(FLAGS_num_categories);
  config.hotness = absl::GetFlag(FLAGS_hotness);
  config.groups_per_bank = absl::GetFlag(FLAGS_groups_per_bank);
  config.batch_size = absl::GetFlag(FLAGS_batch_size);
  config.alpha = absl::GetFlag(FLAGS_alpha);
  config.seed = absl::GetFlag(FLAGS_seed);
  config.shuffle_within_query = absl::GetFlag(FLAGS_shuffle_within_query);

  try {
    cuembed::utils::ValidateGroupedWorkloadConfig(config);
    const auto frequency =
        ParseFrequencyLayout(absl::GetFlag(FLAGS_frequency_layout));
    const auto coaccess =
        ParseCoaccessLayout(absl::GetFlag(FLAGS_coaccess_layout));
    const auto scatter =
        ParseScatterPolicy(absl::GetFlag(FLAGS_scatter_policy));
    const bool half = absl::GetFlag(FLAGS_half_embedding_type);
    const bool int64_indices = absl::GetFlag(FLAGS_use_int64_indices);
    const bool fp16_math = absl::GetFlag(FLAGS_fp16_math);
    if (!half && fp16_math) {
      LOG(FATAL) << "--fp16_math requires --half_embedding_type=true.";
    }
    if (half && int64_indices && fp16_math) {
      RunBenchmark<__half, int64_t, true>(config, frequency, coaccess, scatter);
    } else if (half && !int64_indices && fp16_math) {
      RunBenchmark<__half, int32_t, true>(config, frequency, coaccess, scatter);
    } else if (half && int64_indices) {
      RunBenchmark<__half, int64_t, false>(config, frequency, coaccess, scatter);
    } else if (half) {
      RunBenchmark<__half, int32_t, false>(config, frequency, coaccess, scatter);
    } else if (int64_indices) {
      RunBenchmark<float, int64_t, false>(config, frequency, coaccess, scatter);
    } else {
      RunBenchmark<float, int32_t, false>(config, frequency, coaccess, scatter);
    }
  } catch (const std::exception& error) {
    LOG(ERROR) << error.what();
    return 2;
  }
  return 0;
}
