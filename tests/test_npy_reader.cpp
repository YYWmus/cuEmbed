// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "utils/include/npy_reader.h"

namespace {

template <typename T>
void WriteNpy(const std::filesystem::path& path,
              const std::string& descr,
              const std::string& shape,
              const std::vector<T>& values) {
  std::string header = "{'descr': '" + descr +
                       "', 'fortran_order': False, 'shape': " + shape + ", }";
  const size_t preamble = 10;
  const size_t padding = (64 - ((preamble + header.size() + 1) % 64)) % 64;
  header.append(padding, ' ');
  header.push_back('\n');
  std::ofstream output(path, std::ios::binary);
  const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
  output.write(magic, sizeof(magic));
  const uint16_t length = static_cast<uint16_t>(header.size());
  const char length_bytes[] = {static_cast<char>(length & 0xff),
                               static_cast<char>((length >> 8) & 0xff)};
  output.write(length_bytes, 2);
  output.write(header.data(), header.size());
  output.write(reinterpret_cast<const char*>(values.data()),
               values.size() * sizeof(T));
}

TEST(NpyIntegerReader, ReadsInt32MatrixSlice) {
  const auto path = std::filesystem::temp_directory_path() /
                    "cuembed_npy_reader_int32.npy";
  WriteNpy<int32_t>(path, "<i4", "(2, 3)", {0, 1, 2, 3, 4, 5});
  cuembed::utils::NpyIntegerReader reader(path.string());
  EXPECT_EQ(reader.shape(), (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(reader.Read(2, 3), (std::vector<int64_t>{2, 3, 4}));
  std::filesystem::remove(path);
}

TEST(NpyIntegerReader, ReadsInt64Vector) {
  const auto path = std::filesystem::temp_directory_path() /
                    "cuembed_npy_reader_int64.npy";
  WriteNpy<int64_t>(path, "<i8", "(4,)", {7, 11, 13, 17});
  cuembed::utils::NpyIntegerReader reader(path.string());
  EXPECT_EQ(reader.shape(), (std::vector<int64_t>{4}));
  EXPECT_EQ(reader.Read(1, 2), (std::vector<int64_t>{11, 13}));
  EXPECT_THROW(reader.Read(3, 2), std::out_of_range);
  std::filesystem::remove(path);
}

TEST(NpyIntegerReader, RejectsUnsupportedDtype) {
  const auto path = std::filesystem::temp_directory_path() /
                    "cuembed_npy_reader_float.npy";
  WriteNpy<float>(path, "<f4", "(1,)", {1.0F});
  EXPECT_THROW(cuembed::utils::NpyIntegerReader reader(path.string()),
               std::runtime_error);
  std::filesystem::remove(path);
}

}  // namespace
