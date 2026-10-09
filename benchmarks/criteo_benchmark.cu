// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "utils/include/embedding_allocation.h"
#include "utils/include/embedding_utils.h"
#include "utils/include/npy_reader.h"

ABSL_FLAG(std::string, indices_npy, "", "Fused Criteo sparse IDs with shape (N, 26).");
ABSL_FLAG(int64_t, indices_start_row, 0, "First Criteo sample row to benchmark.");
ABSL_FLAG(std::string, physical_to_logical_npy, "",
          "Optional rank-1 physical-to-logical row mapping.");
ABSL_FLAG(std::string, layout, "identity", "Layout label recorded in CSV output.");
ABSL_FLAG(bool, replay, true,
          "Replay one batch with cache eviction; false streams consecutive batches.");
ABSL_FLAG(int, num_categories, 0, "Number of rows in the fused embedding table.");
ABSL_FLAG(int, embed_width, 16, "Embedding vector width.");
ABSL_FLAG(int, batch_size, 2048, "Samples per lookup invocation.");
ABSL_FLAG(int, iterations, 100, "Number of timed lookup invocations.");
ABSL_FLAG(bool, use_int64_indices, true, "Use int64 rather than int32 lookup indices.");
ABSL_FLAG(bool, half_embedding_type, false, "Use fp16 rather than fp32 embeddings.");
ABSL_FLAG(bool, fp16_math, false, "Accumulate fp16 embeddings using fp16 arithmetic.");
ABSL_FLAG(bool, check_result, false, "Check the last GPU result against a CPU reference.");
ABSL_FLAG(std::string, result_file, "", "Optional raw final-result output file.");
ABSL_FLAG(bool, enable_csv, true, "Append one result row to CSV output.");
ABSL_FLAG(std::string, csv_output, "criteo_benchmark_out.csv", "CSV output path.");
ABSL_FLAG(int64_t, l2_persist_start_row, 0, "First physical row in the persistence window.");
ABSL_FLAG(int64_t, l2_persist_rows, 0, "Rows in the optional L2 persistence window.");
ABSL_FLAG(int64_t, l2_persist_region_bytes, 0,
          "Bytes in the optional L2 persistence window; mutually exclusive with rows.");
ABSL_FLAG(std::string, l2_persist_lifetime, "run",
          "Persistence lifetime: run retains priority across invocations; kernel "
          "resets priority after every lookup before any cache eviction.");

namespace {

constexpr int64_t kCriteoHotness = 26;

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
  void* base_ptr{nullptr};
};

void EnableL2Persistence(const L2PersistenceConfig& config) {
  if (!config.enabled) return;
  cudaStreamAttrValue attribute{};
  attribute.accessPolicyWindow.base_ptr = config.base_ptr;
  attribute.accessPolicyWindow.num_bytes = config.effective_bytes;
  attribute.accessPolicyWindow.hitRatio = config.hit_ratio;
  attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  CHECK_CUDA(cudaStreamSetAttribute(
      0, cudaStreamAttributeAccessPolicyWindow, &attribute));
}

void DisableAndResetL2Persistence(const L2PersistenceConfig& config) {
  if (!config.enabled) return;
  cudaStreamAttrValue attribute{};
  attribute.accessPolicyWindow.num_bytes = 0;
  attribute.accessPolicyWindow.hitRatio = 1.0F;
  attribute.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  CHECK_CUDA(cudaStreamSetAttribute(
      0, cudaStreamAttributeAccessPolicyWindow, &attribute));
  CHECK_CUDA(cudaCtxResetPersistingL2Cache());
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
      LOG(FATAL) << "--l2_persist_region_bytes must contain whole embedding rows.";
    }
    rows = persist_bytes / row_bytes;
  }
  if (rows <= 0 || start_row >= num_categories || rows > num_categories - start_row) {
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
      rows, static_cast<int64_t>(properties.accessPolicyMaxWindowSize) / row_bytes);
  if (effective_rows <= 0) LOG(FATAL) << "Embedding rows exceed the L2 policy window.";
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

  config.enabled = true;
  config.requested_start_row = start_row;
  config.requested_rows = rows;
  config.requested_bytes = rows * row_bytes;
  config.effective_start_row = start_row;
  config.effective_rows = effective_rows;
  config.effective_bytes = effective_bytes;
  config.reserved_bytes = actual_set_aside;
  config.hit_ratio = std::min(
      1.0F, static_cast<float>(actual_set_aside) / effective_bytes);
  config.base_ptr = embedding + static_cast<size_t>(start_row) * embed_width;
  EnableL2Persistence(config);
  return config;
}

void ResetL2Persistence(const L2PersistenceConfig& config) {
  if (!config.enabled) return;
  DisableAndResetL2Persistence(config);
  CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                config.previous_set_aside_bytes));
}

__host__ __device__ float LogicalEmbeddingValue(int64_t logical_row, int column) {
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
  const int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = row_count * embed_width;
  if (element >= total) return;
  const int64_t local_row = element / embed_width;
  const int column = static_cast<int>(element % embed_width);
  const int64_t logical_row = physical_to_logical == nullptr
                                  ? physical_start + local_row
                                  : physical_to_logical[local_row];
  embedding[(physical_start + local_row) * embed_width + column] =
      static_cast<ElemT>(LogicalEmbeddingValue(logical_row, column));
}

template <typename ElemT>
void InitializeEmbedding(thrust::device_vector<ElemT>* embedding,
                         int64_t num_categories,
                         int embed_width,
                         const cuembed::utils::NpyIntegerReader* mapping) {
  constexpr int64_t kChunkRows = 1 << 20;
  thrust::device_vector<int64_t> device_mapping;
  std::vector<uint64_t> seen;
  if (mapping != nullptr) {
    seen.assign(static_cast<size_t>((num_categories + 63) / 64), 0);
  }
  for (int64_t start = 0; start < num_categories; start += kChunkRows) {
    const int64_t rows = std::min(kChunkRows, num_categories - start);
    const int64_t* mapping_ptr = nullptr;
    if (mapping != nullptr) {
      const auto host_values = mapping->Read(start, rows);
      for (const int64_t value : host_values) {
        if (value < 0 || value >= num_categories) {
          LOG(FATAL) << "Physical-to-logical mapping contains an out-of-range row.";
        }
        const uint64_t mask = uint64_t{1} << (value % 64);
        uint64_t& word = seen[static_cast<size_t>(value / 64)];
        if ((word & mask) != 0) {
          LOG(FATAL) << "Physical-to-logical mapping is not a permutation; duplicate logical row "
                     << value << ".";
        }
        word |= mask;
      }
      device_mapping.assign(host_values.begin(), host_values.end());
      mapping_ptr = thrust::raw_pointer_cast(device_mapping.data());
    }
    const int64_t elements = rows * embed_width;
    const int blocks = static_cast<int>((elements + 255) / 256);
    InitializeEmbeddingKernel<<<blocks, 256>>>(
        thrust::raw_pointer_cast(embedding->data()), embed_width, start, rows,
        mapping_ptr);
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaDeviceSynchronize());
}

std::vector<int64_t> ReadCyclicBatch(
    const cuembed::utils::NpyIntegerReader& reader,
    int64_t start_row,
    int64_t batch_size) {
  const int64_t num_rows = reader.shape()[0];
  const int64_t first_rows = std::min(batch_size, num_rows - start_row);
  std::vector<int64_t> result =
      reader.Read(start_row * kCriteoHotness, first_rows * kCriteoHotness);
  if (first_rows < batch_size) {
    auto tail = reader.Read(0, (batch_size - first_rows) * kCriteoHotness);
    result.insert(result.end(), tail.begin(), tail.end());
  }
  return result;
}

template <typename IndexT>
void CopyIndices(const std::vector<int64_t>& values,
                 int64_t num_categories,
                 thrust::device_vector<IndexT>* device_indices) {
  std::vector<IndexT> converted(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    const int64_t value = values[i];
    if (value < 0 || value >= num_categories ||
        value > static_cast<int64_t>(std::numeric_limits<IndexT>::max())) {
      LOG(FATAL) << "Criteo physical row ID is invalid for the configured table/index type: "
                 << value;
    }
    converted[i] = static_cast<IndexT>(value);
  }
  device_indices->assign(converted.begin(), converted.end());
  CHECK_CUDA(cudaDeviceSynchronize());
}

std::vector<int64_t> LogicalRowsForBatch(
    const std::vector<int64_t>& physical_rows,
    const cuembed::utils::NpyIntegerReader* mapping) {
  if (mapping == nullptr) return physical_rows;
  std::vector<int64_t> logical;
  logical.reserve(physical_rows.size());
  for (const int64_t physical : physical_rows) {
    logical.push_back(mapping->Read(physical, 1)[0]);
  }
  return logical;
}

template <typename ElemT>
float ToFloat(ElemT value) {
  return static_cast<float>(value);
}

template <>
float ToFloat(__half value) {
  return __half2float(value);
}

template <typename ElemT>
void CheckResult(const thrust::device_vector<ElemT>& result,
                 const std::vector<int64_t>& physical_rows,
                 const cuembed::utils::NpyIntegerReader* mapping,
                 int batch_size,
                 int embed_width) {
  const auto logical_rows = LogicalRowsForBatch(physical_rows, mapping);
  thrust::host_vector<ElemT> host_result = result;
  const float tolerance = std::is_same_v<ElemT, __half> ? 0.08F : 1e-4F;
  for (int sample = 0; sample < batch_size; ++sample) {
    for (int column = 0; column < embed_width; ++column) {
      float expected = 0.0F;
      for (int hot = 0; hot < kCriteoHotness; ++hot) {
        expected += LogicalEmbeddingValue(
            logical_rows[static_cast<size_t>(sample) * kCriteoHotness + hot],
            column);
      }
      const float actual = ToFloat(
          host_result[static_cast<size_t>(sample) * embed_width + column]);
      if (std::abs(actual - expected) > tolerance) {
        LOG(FATAL) << "Result mismatch at sample " << sample << ", column "
                   << column << ": actual=" << actual
                   << " expected=" << expected;
      }
    }
  }
}

template <typename ElemT>
void WriteResult(const thrust::device_vector<ElemT>& result,
                 const std::string& path) {
  if (path.empty()) return;
  thrust::host_vector<ElemT> host = result;
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) LOG(FATAL) << "Unable to open result file: " << path;
  output.write(reinterpret_cast<const char*>(host.data()),
               static_cast<std::streamsize>(host.size() * sizeof(ElemT)));
  if (!output) LOG(FATAL) << "Unable to write result file: " << path;
}

void DumpCsvHeader(std::ofstream& output) {
  output << "input_source,layout,replay,indices_start_row,num_rows,rows_consumed,"
            "wrap_count,num_categories,batch_size,hotness,embed_width,iterations,"
            "elapsed_time_ms,avg_time_ms,wall_elapsed_ms,avg_wall_ms,"
            "algo_bw_l2,l2_persist_enabled,l2_persist_requested_start_row,"
            "l2_persist_requested_rows,l2_persist_requested_bytes,"
            "l2_persist_effective_start_row,l2_persist_effective_rows,"
            "l2_persist_effective_bytes,l2_persist_reserved_bytes,"
            "l2_persist_hit_ratio,l2_persist_lifetime,l2_persist_reset_count,"
            "cache_clear_bytes\n";
}

bool FileExists(const std::string& path) {
  std::ifstream input(path);
  return input.good();
}

template <typename ElemT, typename IndexT, bool fp16_math>
void RunBenchmark() {
  const std::string indices_path = absl::GetFlag(FLAGS_indices_npy);
  const std::string mapping_path = absl::GetFlag(FLAGS_physical_to_logical_npy);
  const int64_t requested_start = absl::GetFlag(FLAGS_indices_start_row);
  const int64_t num_categories = absl::GetFlag(FLAGS_num_categories);
  const int batch_size = absl::GetFlag(FLAGS_batch_size);
  const int embed_width = absl::GetFlag(FLAGS_embed_width);
  const int iterations = absl::GetFlag(FLAGS_iterations);
  const bool replay = absl::GetFlag(FLAGS_replay);
  const std::string persistence_lifetime =
      absl::GetFlag(FLAGS_l2_persist_lifetime);
  if (persistence_lifetime != "run" && persistence_lifetime != "kernel") {
    LOG(FATAL) << "--l2_persist_lifetime must be run or kernel.";
  }
  const bool kernel_lifetime = persistence_lifetime == "kernel";
  if (indices_path.empty() || num_categories <= 0 || batch_size <= 0 ||
      embed_width <= 0 || iterations <= 0 || requested_start < 0) {
    LOG(FATAL) << "Indices path, dimensions, iterations, and start row must be valid.";
  }

  cuembed::utils::NpyIntegerReader indices(indices_path);
  if (indices.shape().size() != 2 || indices.shape()[1] != kCriteoHotness) {
    LOG(FATAL) << "--indices_npy must have shape (N, 26).";
  }
  const int64_t num_rows = indices.shape()[0];
  if (num_rows <= 0 || batch_size > num_rows || requested_start >= num_rows) {
    LOG(FATAL) << "Start row and batch size must describe a valid cyclic batch.";
  }
  const int64_t start_row = requested_start;

  std::unique_ptr<cuembed::utils::NpyIntegerReader> mapping;
  if (!mapping_path.empty()) {
    mapping = std::make_unique<cuembed::utils::NpyIntegerReader>(mapping_path);
    if (mapping->shape().size() != 1 || mapping->shape()[0] != num_categories) {
      LOG(FATAL) << "Physical-to-logical mapping length must equal num_categories.";
    }
  }

  cuembed::utils::AllocationOptions options;
  options.num_categories(static_cast<int>(num_categories))
      .batch_size(batch_size)
      .hotness(kCriteoHotness)
      .alpha(0.0F)
      .embed_width(embed_width)
      .combine_mode(cuembed::CombineMode::kSum)
      .is_csr(false)
      .is_weighted(false);

  thrust::device_vector<ElemT> embedding(
      static_cast<size_t>(num_categories) * embed_width);
  thrust::device_vector<IndexT> device_indices(
      static_cast<size_t>(batch_size) * kCriteoHotness);
  thrust::device_vector<int> offsets;
  thrust::device_vector<ElemT> weights;
  thrust::device_vector<ElemT> result(
      static_cast<size_t>(batch_size) * embed_width);
  InitializeEmbedding(&embedding, num_categories, embed_width, mapping.get());

  const L2PersistenceConfig persistence = ConfigureL2Persistence(
      thrust::raw_pointer_cast(embedding.data()), num_categories, embed_width,
      absl::GetFlag(FLAGS_l2_persist_start_row),
      absl::GetFlag(FLAGS_l2_persist_rows),
      absl::GetFlag(FLAGS_l2_persist_region_bytes));

  auto run_forward = [&]() {
    cuembed::utils::RunForward<ElemT, IndexT, int, fp16_math>(
        options, embedding, device_indices, offsets, weights, &result);
  };

  std::vector<int64_t> last_batch;
  if (replay) {
    last_batch = ReadCyclicBatch(indices, start_row, batch_size);
  } else {
    const int64_t predecessor =
        (start_row + num_rows - (batch_size % num_rows)) % num_rows;
    last_batch = ReadCyclicBatch(indices, predecessor, batch_size);
  }
  CopyIndices(last_batch, num_categories, &device_indices);
  run_forward();
  CHECK_CUDA(cudaDeviceSynchronize());

  thrust::device_vector<int> eviction_buffer;
  volatile int eviction_sink = 0;
  size_t cache_clear_bytes = 0;
  int64_t persistence_reset_count = 0;
  if (kernel_lifetime && persistence.enabled) {
    DisableAndResetL2Persistence(persistence);
    ++persistence_reset_count;
  }
  if (replay) {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    const size_t values = std::max<size_t>(
        1, (static_cast<size_t>(properties.l2CacheSize) * 2 + sizeof(int) - 1) /
               sizeof(int));
    cache_clear_bytes = values * sizeof(int);
    eviction_buffer.assign(values, 1);
    eviction_sink += thrust::reduce(eviction_buffer.begin(), eviction_buffer.end(), 0);
  }
  if (kernel_lifetime && persistence.enabled) EnableL2Persistence(persistence);

  cudaEvent_t start_event, stop_event;
  CHECK_CUDA(cudaEventCreate(&start_event));
  CHECK_CUDA(cudaEventCreate(&stop_event));
  float kernel_elapsed_ms = 0.0F;
  const auto wall_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (!replay) {
      const int64_t iteration_start =
          (start_row + static_cast<int64_t>(iteration) * batch_size) % num_rows;
      last_batch = ReadCyclicBatch(indices, iteration_start, batch_size);
      CopyIndices(last_batch, num_categories, &device_indices);
    }
    CHECK_CUDA(cudaEventRecord(start_event));
    run_forward();
    CHECK_CUDA(cudaEventRecord(stop_event));
    CHECK_CUDA(cudaEventSynchronize(stop_event));
    float iteration_ms = 0.0F;
    CHECK_CUDA(cudaEventElapsedTime(&iteration_ms, start_event, stop_event));
    kernel_elapsed_ms += iteration_ms;
    if (kernel_lifetime && persistence.enabled) {
      DisableAndResetL2Persistence(persistence);
      ++persistence_reset_count;
    }
    if (replay && iteration + 1 < iterations) {
      eviction_sink += thrust::reduce(eviction_buffer.begin(), eviction_buffer.end(), 0);
    }
    if (kernel_lifetime && persistence.enabled && iteration + 1 < iterations) {
      EnableL2Persistence(persistence);
    }
  }
  const auto wall_stop = std::chrono::steady_clock::now();
  const double wall_elapsed_ms =
      std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
  if (eviction_sink == std::numeric_limits<int>::min()) std::cerr << eviction_sink;

  if (absl::GetFlag(FLAGS_check_result)) {
    CheckResult(result, last_batch, mapping.get(), batch_size, embed_width);
  }
  WriteResult(result, absl::GetFlag(FLAGS_result_file));

  const int64_t rows_consumed = static_cast<int64_t>(batch_size) * iterations;
  const int64_t wrap_count = replay
      ? ((start_row + batch_size > num_rows) ? iterations : 0)
      : (start_row + rows_consumed - 1) / num_rows;
  const double bytes = static_cast<double>(sizeof(ElemT)) * rows_consumed *
                       (kCriteoHotness + 1) * embed_width;
  const double algo_bw = bytes / (kernel_elapsed_ms * 1.0e6);

  if (absl::GetFlag(FLAGS_enable_csv)) {
    const std::string csv_path = absl::GetFlag(FLAGS_csv_output);
    const bool exists = FileExists(csv_path);
    std::ofstream output(csv_path, std::ios::app);
    if (!output) LOG(FATAL) << "Unable to open CSV output: " << csv_path;
    if (!exists) DumpCsvHeader(output);
    output << "criteo," << absl::GetFlag(FLAGS_layout) << "," << replay << ","
           << start_row << "," << num_rows << "," << rows_consumed << ","
           << wrap_count << "," << num_categories << "," << batch_size << ","
           << kCriteoHotness << "," << embed_width << "," << iterations << ","
           << kernel_elapsed_ms << "," << kernel_elapsed_ms / iterations << ","
           << wall_elapsed_ms << "," << wall_elapsed_ms / iterations << ","
           << algo_bw << "," << persistence.enabled << ","
           << persistence.requested_start_row << "," << persistence.requested_rows << ","
           << persistence.requested_bytes << "," << persistence.effective_start_row << ","
           << persistence.effective_rows << "," << persistence.effective_bytes << ","
           << persistence.reserved_bytes << "," << persistence.hit_ratio << ","
           << persistence_lifetime << "," << persistence_reset_count << ","
           << cache_clear_bytes << "\n";
  }

  LOG(INFO) << "Criteo forward: replay=" << replay
            << " iterations=" << iterations
            << " avg kernel ms=" << kernel_elapsed_ms / iterations
            << " avg wall ms=" << wall_elapsed_ms / iterations
            << " application BW [GB/s]=" << algo_bw
            << " l2 persistence lifetime=" << persistence_lifetime;
  CHECK_CUDA(cudaEventDestroy(start_event));
  CHECK_CUDA(cudaEventDestroy(stop_event));
  if (kernel_lifetime && persistence.enabled) {
    CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                  persistence.previous_set_aside_bytes));
  } else {
    ResetL2Persistence(persistence);
  }
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  const bool half = absl::GetFlag(FLAGS_half_embedding_type);
  const bool int64_indices = absl::GetFlag(FLAGS_use_int64_indices);
  const bool fp16_math = absl::GetFlag(FLAGS_fp16_math);
  if (!half && fp16_math) LOG(FATAL) << "--fp16_math requires fp16 embeddings.";
  if (half && int64_indices && fp16_math) RunBenchmark<__half, int64_t, true>();
  else if (half && !int64_indices && fp16_math) RunBenchmark<__half, int32_t, true>();
  else if (half && int64_indices) RunBenchmark<__half, int64_t, false>();
  else if (half) RunBenchmark<__half, int32_t, false>();
  else if (int64_indices) RunBenchmark<float, int64_t, false>();
  else RunBenchmark<float, int32_t, false>();
  return 0;
}
