// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef UTILS_INCLUDE_NPY_READER_H_
#define UTILS_INCLUDE_NPY_READER_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuembed {
namespace utils {

// A deliberately small reader for the contiguous, C-order integer .npy files
// produced by NumPy in this repository. It reads slices without materializing
// a full Criteo day in host memory.
class NpyIntegerReader {
 public:
  explicit NpyIntegerReader(const std::string& path) : path_(path) {
    ParseHeader();
  }

  const std::string& path() const { return path_; }
  const std::vector<int64_t>& shape() const { return shape_; }
  int element_bytes() const { return element_bytes_; }
  int64_t element_count() const { return element_count_; }

  std::vector<int64_t> Read(int64_t start, int64_t count) const {
    if (start < 0 || count < 0 || start > element_count_ - count) {
      throw std::out_of_range("Requested .npy slice is outside " + path_);
    }
    std::ifstream input(path_, std::ios::binary);
    if (!input) throw std::runtime_error("Unable to open .npy file: " + path_);
    const uint64_t byte_offset =
        data_offset_ + static_cast<uint64_t>(start) * element_bytes_;
    input.seekg(static_cast<std::streamoff>(byte_offset));
    if (!input) throw std::runtime_error("Unable to seek in .npy file: " + path_);

    std::vector<char> bytes(static_cast<size_t>(count) * element_bytes_);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
      throw std::runtime_error("Short read from .npy file: " + path_);
    }

    std::vector<int64_t> values(static_cast<size_t>(count));
    if (element_bytes_ == 4) {
      for (int64_t i = 0; i < count; ++i) {
        int32_t value;
        std::memcpy(&value, bytes.data() + i * 4, 4);
        values[static_cast<size_t>(i)] = value;
      }
    } else {
      for (int64_t i = 0; i < count; ++i) {
        int64_t value;
        std::memcpy(&value, bytes.data() + i * 8, 8);
        values[static_cast<size_t>(i)] = value;
      }
    }
    return values;
  }

 private:
  static uint16_t ReadU16(std::istream& input) {
    unsigned char bytes[2];
    input.read(reinterpret_cast<char*>(bytes), 2);
    return static_cast<uint16_t>(bytes[0]) |
           (static_cast<uint16_t>(bytes[1]) << 8);
  }

  static uint32_t ReadU32(std::istream& input) {
    unsigned char bytes[4];
    input.read(reinterpret_cast<char*>(bytes), 4);
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) |
           (static_cast<uint32_t>(bytes[3]) << 24);
  }

  void ParseHeader() {
    std::ifstream input(path_, std::ios::binary);
    if (!input) throw std::runtime_error("Unable to open .npy file: " + path_);
    char magic[6];
    input.read(magic, 6);
    const char expected[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    if (!input || std::memcmp(magic, expected, 6) != 0) {
      throw std::runtime_error("Not a NumPy .npy file: " + path_);
    }
    unsigned char version[2];
    input.read(reinterpret_cast<char*>(version), 2);
    if (!input || version[0] == 0 || version[0] > 3) {
      throw std::runtime_error("Unsupported .npy version in " + path_);
    }
    const uint32_t header_length =
        version[0] == 1 ? ReadU16(input) : ReadU32(input);
    std::string header(header_length, '\0');
    input.read(header.data(), header_length);
    if (!input) throw std::runtime_error("Truncated .npy header: " + path_);
    data_offset_ = static_cast<uint64_t>(input.tellg());

    std::smatch match;
    const std::regex descr_re("['\\\"]descr['\\\"]\\s*:\\s*['\\\"]([^'\\\"]+)['\\\"]");
    if (!std::regex_search(header, match, descr_re)) {
      throw std::runtime_error("Missing dtype in .npy header: " + path_);
    }
    const std::string descr = match[1].str();
    if (descr == ">i4" || descr == ">i8") {
      throw std::runtime_error("Big-endian .npy integers are unsupported: " + path_);
    }
    if (descr == "<i4" || descr == "|i4" || descr == "=i4") {
      element_bytes_ = 4;
    } else if (descr == "<i8" || descr == "|i8" || descr == "=i8") {
      element_bytes_ = 8;
    } else {
      throw std::runtime_error("Expected int32 or int64 .npy dtype, got " + descr);
    }

    const std::regex fortran_re("['\\\"]fortran_order['\\\"]\\s*:\\s*(True|False)");
    if (!std::regex_search(header, match, fortran_re) || match[1] != "False") {
      throw std::runtime_error("Only C-order .npy arrays are supported: " + path_);
    }
    const std::regex shape_re("['\\\"]shape['\\\"]\\s*:\\s*\\(([^)]*)\\)");
    if (!std::regex_search(header, match, shape_re)) {
      throw std::runtime_error("Missing shape in .npy header: " + path_);
    }
    const std::string shape_text = match[1].str();
    const std::regex dim_re("([0-9]+)");
    for (std::sregex_iterator it(shape_text.begin(), shape_text.end(), dim_re), end;
         it != end; ++it) {
      shape_.push_back(std::stoll((*it)[1].str()));
    }
    if (shape_.empty()) throw std::runtime_error("Empty .npy shape: " + path_);
    element_count_ = 1;
    for (const int64_t dim : shape_) {
      if (dim < 0 || (dim != 0 && element_count_ >
                                      std::numeric_limits<int64_t>::max() / dim)) {
        throw std::runtime_error("Invalid .npy shape: " + path_);
      }
      element_count_ *= dim;
    }
  }

  std::string path_;
  std::vector<int64_t> shape_;
  int element_bytes_{0};
  int64_t element_count_{0};
  uint64_t data_offset_{0};
};

}  // namespace utils
}  // namespace cuembed

#endif  // UTILS_INCLUDE_NPY_READER_H_
