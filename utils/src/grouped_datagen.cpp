// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "utils/include/grouped_datagen.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace cuembed {
namespace utils {
namespace {

constexpr uint64_t kTraceDomain = 0x74726163655f726eULL;
constexpr uint64_t kOrderDomain = 0x6f726465725f726eULL;
constexpr uint64_t kLayoutDomain = 0x6c61796f75745f72ULL;

uint64_t MixSeed(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

std::mt19937_64 MakeRng(uint64_t seed, uint64_t domain) {
  return std::mt19937_64(MixSeed(seed ^ domain));
}

int64_t SampleBank(int64_t num_banks,
                   double alpha,
                   std::mt19937_64* rng) {
  if (alpha == 0.0) {
    return std::uniform_int_distribution<int64_t>(0, num_banks - 1)(*rng);
  }
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  const double gamma = 1.0 - alpha;
  const double x = uniform(*rng);
  const double translated = std::pow(
      x * (std::pow(static_cast<double>(num_banks + 1), gamma) - 1.0) +
          1.0,
      1.0 / gamma);
  const int64_t one_based = static_cast<int64_t>(translated);
  return std::clamp<int64_t>(one_based - 1, 0, num_banks - 1);
}

void AppendGroup(std::vector<int64_t>* physical_to_logical,
                 int64_t group,
                 int hotness) {
  const int64_t first = group * hotness;
  for (int member = 0; member < hotness; ++member) {
    physical_to_logical->push_back(first + member);
  }
}

}  // namespace

void ValidateGroupedWorkloadConfig(const GroupedWorkloadConfig& config) {
  if (config.num_categories <= 0 || config.hotness < 2 ||
      config.groups_per_bank < 2 || config.batch_size <= 0) {
    throw std::invalid_argument(
        "num_categories and batch_size must be positive; hotness and "
        "groups_per_bank must be at least 2");
  }
  const int64_t rows_per_bank =
      static_cast<int64_t>(config.hotness) * config.groups_per_bank;
  if (config.num_categories % rows_per_bank != 0) {
    throw std::invalid_argument(
        "num_categories must be divisible by hotness*groups_per_bank");
  }
  if (config.num_categories / rows_per_bank < 2) {
    throw std::invalid_argument("the workload must contain at least 2 banks");
  }
  if (config.alpha < 0.0 || config.alpha == 1.0 ||
      !std::isfinite(config.alpha)) {
    throw std::invalid_argument(
        "alpha must be 0 or a finite positive value other than 1");
  }
}

uint64_t StableChecksum(const std::vector<int64_t>& values) {
  uint64_t hash = 1469598103934665603ULL;
  for (const int64_t value : values) {
    const uint64_t encoded = static_cast<uint64_t>(value);
    for (int byte = 0; byte < 8; ++byte) {
      hash ^= (encoded >> (byte * 8)) & 0xffULL;
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

std::string ChecksumHex(uint64_t checksum) {
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << checksum;
  return output.str();
}

GroupedWorkload GenerateGroupedWorkload(const GroupedWorkloadConfig& config) {
  ValidateGroupedWorkloadConfig(config);
  GroupedWorkload workload;
  workload.num_groups = config.num_categories / config.hotness;
  workload.num_frequency_banks =
      workload.num_groups / config.groups_per_bank;
  workload.group_trace.reserve(config.batch_size);
  workload.logical_indices.reserve(
      static_cast<size_t>(config.batch_size) * config.hotness);

  auto trace_rng = MakeRng(config.seed, kTraceDomain);
  auto order_rng = MakeRng(config.seed, kOrderDomain);
  std::uniform_int_distribution<int> group_in_bank(
      0, config.groups_per_bank - 1);
  std::vector<int> members(config.hotness);
  std::iota(members.begin(), members.end(), 0);
  for (int sample = 0; sample < config.batch_size; ++sample) {
    const int64_t bank =
        SampleBank(workload.num_frequency_banks, config.alpha, &trace_rng);
    const int64_t group =
        bank * config.groups_per_bank + group_in_bank(trace_rng);
    workload.group_trace.push_back(group);
    std::iota(members.begin(), members.end(), 0);
    if (config.shuffle_within_query) {
      std::shuffle(members.begin(), members.end(), order_rng);
    }
    for (const int member : members) {
      workload.logical_indices.push_back(group * config.hotness + member);
    }
  }
  workload.trace_checksum = StableChecksum(workload.group_trace);
  return workload;
}

GroupedLayoutMapping BuildGroupedLayoutMapping(
    const GroupedWorkloadConfig& config,
    FrequencyLayout frequency_layout,
    CoaccessLayout coaccess_layout,
    ScatterPolicy scatter_policy) {
  ValidateGroupedWorkloadConfig(config);
  const int64_t num_groups = config.num_categories / config.hotness;
  const int64_t num_banks = num_groups / config.groups_per_bank;
  std::vector<int64_t> physical_to_logical;
  physical_to_logical.reserve(config.num_categories);
  auto layout_rng = MakeRng(config.seed, kLayoutDomain);

  if (frequency_layout == FrequencyLayout::kSorted &&
      coaccess_layout == CoaccessLayout::kCompact) {
    physical_to_logical.resize(config.num_categories);
    std::iota(physical_to_logical.begin(), physical_to_logical.end(), 0);
  } else if (frequency_layout == FrequencyLayout::kSorted &&
             scatter_policy == ScatterPolicy::kInterleaved) {
    for (int64_t bank = 0; bank < num_banks; ++bank) {
      for (int member = 0; member < config.hotness; ++member) {
        for (int in_bank = 0; in_bank < config.groups_per_bank; ++in_bank) {
          const int64_t group = bank * config.groups_per_bank + in_bank;
          physical_to_logical.push_back(group * config.hotness + member);
        }
      }
    }
  } else if (frequency_layout == FrequencyLayout::kSorted) {
    const int64_t rows_per_bank =
        static_cast<int64_t>(config.hotness) * config.groups_per_bank;
    for (int64_t bank = 0; bank < num_banks; ++bank) {
      std::vector<int64_t> rows(rows_per_bank);
      std::iota(rows.begin(), rows.end(), bank * rows_per_bank);
      std::shuffle(rows.begin(), rows.end(), layout_rng);
      physical_to_logical.insert(
          physical_to_logical.end(), rows.begin(), rows.end());
    }
  } else if (coaccess_layout == CoaccessLayout::kCompact) {
    std::vector<int64_t> groups(num_groups);
    std::iota(groups.begin(), groups.end(), 0);
    std::shuffle(groups.begin(), groups.end(), layout_rng);
    for (const int64_t group : groups) {
      AppendGroup(&physical_to_logical, group, config.hotness);
    }
  } else if (scatter_policy == ScatterPolicy::kInterleaved) {
    std::vector<int64_t> groups(num_groups);
    std::iota(groups.begin(), groups.end(), 0);
    std::shuffle(groups.begin(), groups.end(), layout_rng);
    for (int member = 0; member < config.hotness; ++member) {
      for (const int64_t group : groups) {
        physical_to_logical.push_back(group * config.hotness + member);
      }
    }
  } else {
    physical_to_logical.resize(config.num_categories);
    std::iota(physical_to_logical.begin(), physical_to_logical.end(), 0);
    std::shuffle(
        physical_to_logical.begin(), physical_to_logical.end(), layout_rng);
  }

  if (physical_to_logical.size() !=
      static_cast<size_t>(config.num_categories)) {
    throw std::logic_error("grouped layout did not assign every physical row");
  }
  GroupedLayoutMapping mapping;
  mapping.physical_to_logical = std::move(physical_to_logical);
  mapping.logical_to_physical.resize(config.num_categories, -1);
  for (int64_t physical = 0; physical < config.num_categories; ++physical) {
    const int64_t logical = mapping.physical_to_logical[physical];
    if (logical < 0 || logical >= config.num_categories ||
        mapping.logical_to_physical[logical] != -1) {
      throw std::logic_error("grouped layout is not a permutation");
    }
    mapping.logical_to_physical[logical] = physical;
  }
  mapping.mapping_checksum = StableChecksum(mapping.logical_to_physical);
  int64_t adjacent_pairs = 0;
  int64_t span_sum = 0;
  for (int64_t group = 0; group < num_groups; ++group) {
    std::vector<int64_t> positions(config.hotness);
    for (int member = 0; member < config.hotness; ++member) {
      positions[member] = mapping.logical_to_physical[
          group * config.hotness + member];
    }
    std::sort(positions.begin(), positions.end());
    for (int member = 1; member < config.hotness; ++member) {
      if (positions[member] == positions[member - 1] + 1) {
        ++adjacent_pairs;
      }
    }
    const int64_t span = positions.back() - positions.front() + 1;
    span_sum += span;
    if (span == config.hotness) ++mapping.contiguous_groups;
  }
  mapping.adjacent_member_pair_ratio =
      static_cast<double>(adjacent_pairs) /
      static_cast<double>(num_groups * (config.hotness - 1));
  mapping.mean_group_span_rows =
      static_cast<double>(span_sum) / static_cast<double>(num_groups);
  return mapping;
}

}  // namespace utils
}  // namespace cuembed
