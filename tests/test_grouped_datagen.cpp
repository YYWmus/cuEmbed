// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "utils/include/grouped_datagen.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

#include "gtest/gtest.h"

namespace {

using cuembed::utils::BuildGroupedLayoutMapping;
using cuembed::utils::CoaccessLayout;
using cuembed::utils::FrequencyLayout;
using cuembed::utils::GenerateGroupedWorkload;
using cuembed::utils::GroupedWorkloadConfig;
using cuembed::utils::ScatterPolicy;

GroupedWorkloadConfig TestConfig() {
  GroupedWorkloadConfig config;
  config.num_categories = 48;
  config.hotness = 3;
  config.groups_per_bank = 2;
  config.batch_size = 256;
  config.alpha = 1.2;
  config.seed = 123456;
  config.shuffle_within_query = true;
  return config;
}

TEST(GroupedDatagen, RejectsInvalidDimensionsAndAlpha) {
  auto config = TestConfig();
  config.num_categories = 47;
  EXPECT_THROW(cuembed::utils::ValidateGroupedWorkloadConfig(config),
               std::invalid_argument);
  config = TestConfig();
  config.alpha = 1.0;
  EXPECT_THROW(cuembed::utils::ValidateGroupedWorkloadConfig(config),
               std::invalid_argument);
}

TEST(GroupedDatagen, QueriesContainOneCompleteExclusiveGroup) {
  const auto config = TestConfig();
  const auto workload = GenerateGroupedWorkload(config);
  ASSERT_EQ(workload.group_trace.size(), config.batch_size);
  ASSERT_EQ(workload.logical_indices.size(),
            static_cast<size_t>(config.batch_size * config.hotness));
  std::vector<int64_t> row_counts(config.num_categories, 0);
  for (int sample = 0; sample < config.batch_size; ++sample) {
    const int64_t expected_group = workload.group_trace[sample];
    std::set<int64_t> members;
    for (int hot = 0; hot < config.hotness; ++hot) {
      const int64_t logical =
          workload.logical_indices[sample * config.hotness + hot];
      EXPECT_EQ(logical / config.hotness, expected_group);
      members.insert(logical % config.hotness);
      ++row_counts[logical];
    }
    EXPECT_EQ(members.size(), config.hotness);
  }
  for (int64_t group = 0; group < workload.num_groups; ++group) {
    const int64_t expected = row_counts[group * config.hotness];
    for (int member = 1; member < config.hotness; ++member) {
      EXPECT_EQ(row_counts[group * config.hotness + member], expected);
    }
  }
}

TEST(GroupedDatagen, RngStreamsKeepTraceIndependentOfQueryOrder) {
  auto shuffled = TestConfig();
  auto fixed = shuffled;
  fixed.shuffle_within_query = false;
  const auto shuffled_workload = GenerateGroupedWorkload(shuffled);
  const auto fixed_workload = GenerateGroupedWorkload(fixed);
  EXPECT_EQ(shuffled_workload.group_trace, fixed_workload.group_trace);
  EXPECT_EQ(shuffled_workload.trace_checksum,
            fixed_workload.trace_checksum);
  EXPECT_NE(shuffled_workload.logical_indices,
            fixed_workload.logical_indices);
}

TEST(GroupedDatagen, StableChecksumHasFixedEncoding) {
  EXPECT_EQ(cuembed::utils::StableChecksum({0, 1}),
            0x842360170b01e222ULL);
  EXPECT_EQ(cuembed::utils::ChecksumHex(0x12ULL), "0000000000000012");
}

TEST(GroupedDatagen, EveryLayoutIsADeterministicBijection) {
  const auto config = TestConfig();
  for (const auto frequency : {FrequencyLayout::kSorted,
                               FrequencyLayout::kRandom}) {
    for (const auto coaccess : {CoaccessLayout::kCompact,
                                CoaccessLayout::kScattered}) {
      for (const auto scatter : {ScatterPolicy::kInterleaved,
                                 ScatterPolicy::kRandom}) {
        if (coaccess == CoaccessLayout::kCompact &&
            scatter == ScatterPolicy::kRandom) {
          continue;
        }
        const auto mapping = BuildGroupedLayoutMapping(
            config, frequency, coaccess, scatter);
        const auto repeated = BuildGroupedLayoutMapping(
            config, frequency, coaccess, scatter);
        EXPECT_EQ(mapping.logical_to_physical,
                  repeated.logical_to_physical);
        EXPECT_EQ(mapping.mapping_checksum, repeated.mapping_checksum);
        std::set<int64_t> physical(mapping.logical_to_physical.begin(),
                                   mapping.logical_to_physical.end());
        EXPECT_EQ(physical.size(), config.num_categories);
        EXPECT_EQ(*physical.begin(), 0);
        EXPECT_EQ(*physical.rbegin(), config.num_categories - 1);
        for (int64_t logical = 0; logical < config.num_categories;
             ++logical) {
          EXPECT_EQ(mapping.physical_to_logical[
                        mapping.logical_to_physical[logical]],
                    logical);
        }
      }
    }
  }
}

TEST(GroupedDatagen, CompactLayoutsKeepEveryGroupContiguous) {
  const auto config = TestConfig();
  for (const auto frequency : {FrequencyLayout::kSorted,
                               FrequencyLayout::kRandom}) {
    const auto mapping = BuildGroupedLayoutMapping(
        config, frequency, CoaccessLayout::kCompact,
        ScatterPolicy::kInterleaved);
    const int64_t groups = config.num_categories / config.hotness;
    EXPECT_EQ(mapping.contiguous_groups, groups);
    EXPECT_DOUBLE_EQ(mapping.adjacent_member_pair_ratio, 1.0);
    EXPECT_DOUBLE_EQ(mapping.mean_group_span_rows, config.hotness);
    for (int64_t group = 0; group < groups; ++group) {
      std::vector<int64_t> positions;
      for (int member = 0; member < config.hotness; ++member) {
        positions.push_back(mapping.logical_to_physical[
            group * config.hotness + member]);
      }
      std::sort(positions.begin(), positions.end());
      for (int member = 1; member < config.hotness; ++member) {
        EXPECT_EQ(positions[member], positions[0] + member);
      }
    }
  }
}

TEST(GroupedDatagen, SortedLayoutsKeepRowsInsideTheirFrequencyBank) {
  const auto config = TestConfig();
  const int64_t rows_per_bank =
      static_cast<int64_t>(config.hotness) * config.groups_per_bank;
  for (const auto scatter : {ScatterPolicy::kInterleaved,
                             ScatterPolicy::kRandom}) {
    const auto mapping = BuildGroupedLayoutMapping(
        config, FrequencyLayout::kSorted, CoaccessLayout::kScattered,
        scatter);
    for (int64_t physical = 0; physical < config.num_categories;
         ++physical) {
      const int64_t logical = mapping.physical_to_logical[physical];
      EXPECT_EQ(physical / rows_per_bank, logical / rows_per_bank);
    }
  }
}

TEST(GroupedDatagen, InterleavingUsesExpectedMemberStride) {
  const auto config = TestConfig();
  const int64_t groups = config.num_categories / config.hotness;
  for (const auto frequency : {FrequencyLayout::kSorted,
                               FrequencyLayout::kRandom}) {
    const auto mapping = BuildGroupedLayoutMapping(
        config, frequency, CoaccessLayout::kScattered,
        ScatterPolicy::kInterleaved);
    const int64_t expected_stride =
        frequency == FrequencyLayout::kSorted ? config.groups_per_bank
                                               : groups;
    EXPECT_EQ(mapping.contiguous_groups, 0);
    EXPECT_DOUBLE_EQ(mapping.adjacent_member_pair_ratio, 0.0);
    for (int64_t group = 0; group < groups; ++group) {
      for (int member = 1; member < config.hotness; ++member) {
        EXPECT_EQ(mapping.logical_to_physical[group * config.hotness + member],
                  mapping.logical_to_physical[group * config.hotness] +
                      member * expected_stride);
      }
    }
  }
}

TEST(GroupedDatagen, EveryPhysicalLayoutRecoversTheSameLogicalQueries) {
  const auto config = TestConfig();
  const auto workload = GenerateGroupedWorkload(config);
  for (const auto frequency : {FrequencyLayout::kSorted,
                               FrequencyLayout::kRandom}) {
    for (const auto coaccess : {CoaccessLayout::kCompact,
                                CoaccessLayout::kScattered}) {
      for (const auto scatter : {ScatterPolicy::kInterleaved,
                                 ScatterPolicy::kRandom}) {
        if (coaccess == CoaccessLayout::kCompact &&
            scatter == ScatterPolicy::kRandom) {
          continue;
        }
        const auto mapping = BuildGroupedLayoutMapping(
            config, frequency, coaccess, scatter);
        std::vector<int64_t> recovered;
        recovered.reserve(workload.logical_indices.size());
        for (const int64_t logical : workload.logical_indices) {
          const int64_t physical = mapping.logical_to_physical[logical];
          recovered.push_back(mapping.physical_to_logical[physical]);
        }
        EXPECT_EQ(recovered, workload.logical_indices);
      }
    }
  }
}

}  // namespace
