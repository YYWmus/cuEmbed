// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef UTILS_INCLUDE_GROUPED_DATAGEN_H_
#define UTILS_INCLUDE_GROUPED_DATAGEN_H_

#include <cstdint>
#include <string>
#include <vector>

namespace cuembed {
namespace utils {

enum class FrequencyLayout { kSorted, kRandom };
enum class CoaccessLayout { kCompact, kScattered };
enum class ScatterPolicy { kInterleaved, kRandom };

struct GroupedWorkloadConfig {
  int64_t num_categories{0};
  int hotness{0};
  int groups_per_bank{0};
  int batch_size{0};
  double alpha{0.0};
  uint64_t seed{123456};
  bool shuffle_within_query{true};
};

struct GroupedWorkload {
  int64_t num_groups{0};
  int64_t num_frequency_banks{0};
  std::vector<int64_t> group_trace;
  std::vector<int64_t> logical_indices;
  uint64_t trace_checksum{0};
};

struct GroupedLayoutMapping {
  std::vector<int64_t> logical_to_physical;
  std::vector<int64_t> physical_to_logical;
  uint64_t mapping_checksum{0};
  int64_t contiguous_groups{0};
  double adjacent_member_pair_ratio{0.0};
  double mean_group_span_rows{0.0};
};

void ValidateGroupedWorkloadConfig(const GroupedWorkloadConfig& config);

GroupedWorkload GenerateGroupedWorkload(const GroupedWorkloadConfig& config);

GroupedLayoutMapping BuildGroupedLayoutMapping(
    const GroupedWorkloadConfig& config,
    FrequencyLayout frequency_layout,
    CoaccessLayout coaccess_layout,
    ScatterPolicy scatter_policy);

uint64_t StableChecksum(const std::vector<int64_t>& values);
std::string ChecksumHex(uint64_t checksum);

}  // namespace utils
}  // namespace cuembed

#endif  // UTILS_INCLUDE_GROUPED_DATAGEN_H_
