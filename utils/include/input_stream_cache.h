// SPDX-License-Identifier: Apache-2.0
#ifndef CUEMBED_INPUT_STREAM_CACHE_H_
#define CUEMBED_INPUT_STREAM_CACHE_H_

// CPU-only storage for deterministic streams. Each record is one batch plus
// CRC32C. Native encodings are safe because the descriptor identifies the ABI.
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuembed {
namespace utils {
namespace stream_cache {

inline void Require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error("Input stream cache: " + message);
}

inline uint32_t Crc32c(const void* data, size_t bytes) {
  static const auto table = [] {
    std::array<uint32_t, 256> result{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t value = i;
      for (int bit = 0; bit < 8; ++bit)
        value = (value >> 1) ^ ((value & 1) ? 0x82f63b78U : 0);
      result[i] = value;
    }
    return result;
  }();
  uint32_t crc = ~uint32_t(0);
  const auto* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; ++i)
    crc = table[(crc ^ p[i]) & 255] ^ (crc >> 8);
  return ~crc;
}

// FNV chooses a filename only; the complete descriptor is checked on every hit.
inline std::string Key(const std::string& descriptor) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char c : descriptor) hash = (hash ^ c) * 1099511628211ULL;
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

template <typename IndexT, typename OffsetT>
std::string Descriptor(int rows, int batch, int hotness, float alpha,
                       bool shuffle, bool permute, bool csr,
                       const std::vector<OffsetT>& offsets) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  // Bump generator-v1 whenever datagen or stream generation changes.
  out << "stream-v1;generator-v1;";
#ifdef __GLIBCXX__
  out << "libstdc++=" << __GLIBCXX__ << ';';
#endif
#ifdef _LIBCPP_VERSION
  out << "libc++=" << _LIBCPP_VERSION << ';';
#endif
  uint16_t endian = 1;
  out << "endian=" << int(*reinterpret_cast<unsigned char*>(&endian))
      << ";engine=" << std::default_random_engine::min() << ':'
      << std::default_random_engine::max() << ':' << sizeof(std::default_random_engine)
      << ";seed=" << std::default_random_engine::default_seed
      << ";rows=" << rows << ";batch=" << batch << ";hotness=" << hotness
      << ";alpha=" << std::hexfloat << alpha << std::defaultfloat
      << ";index_bytes=" << sizeof(IndexT) << ";shuffle=" << shuffle
      << ";permute=" << permute << ";csr=" << csr;
  if (csr) {
    out << ";offsets=";
    for (const auto offset : offsets) out << offset << ',';
  }
  return out.str();
}

class Lock {
 public:
  explicit Lock(const std::string& path) {
    fd_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
    Require(fd_ >= 0, "cannot open lock " + path + ": " + std::strerror(errno));
    int status;
    do { status = ::flock(fd_, LOCK_EX); } while (status < 0 && errno == EINTR);
    if (status < 0) {
      ::close(fd_);
      throw std::runtime_error("Input stream cache: cannot lock " + path);
    }
  }
  ~Lock() { ::close(fd_); }
  Lock(const Lock&) = delete;
  Lock& operator=(const Lock&) = delete;
 private:
  int fd_;
};

class TemporaryFile {
 public:
  explicit TemporaryFile(const std::string& path) {
    std::vector<char> name(path.begin(), path.end());
    const std::string suffix = ".tmp.XXXXXX";
    name.insert(name.end(), suffix.begin(), suffix.end());
    name.push_back('\0');
    const int fd = ::mkstemp(name.data());
    Require(fd >= 0, "cannot create temporary file for " + path);
    ::close(fd);
    path_ = name.data();
  }
  ~TemporaryFile() { ::unlink(path_.c_str()); }
  const std::string& path() const { return path_; }
 private:
  std::string path_;
};

template <typename T>
void Write(std::ostream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
template <typename T>
T Read(std::istream& in, const std::string& path) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  Require(bool(in), "truncated cache " + path + "; remove this .bin file and retry");
  return value;
}

struct Result {
  std::string status;
  std::string path;
};

struct Workload {
  int warmup_iterations;
  int iterations;
  int64_t warmup_queries;
  int64_t measurement_queries;
};

inline Workload ResolveWorkload(int batch, int reference_batch, int warmup,
                                int iterations, bool profile_first_batch = false) {
  Require(batch > 0 && reference_batch >= 0 && warmup >= 0 && iterations > 0,
          "invalid streaming workload counts");
  if (reference_batch == 0) reference_batch = batch;
  Require(reference_batch >= batch, "stream reference batch must be >= batch size");
  const int64_t warmup_queries = int64_t(reference_batch) * warmup;
  const int64_t measurement_queries = int64_t(reference_batch) * iterations;
  Require(warmup_queries % batch == 0 && measurement_queries % batch == 0,
          "stream warmup and measurement query boundaries must be exactly divisible by batch size");
  Require(warmup_queries / batch <= std::numeric_limits<int>::max() &&
              measurement_queries / batch <= std::numeric_limits<int>::max(),
          "scaled streaming iterations exceed int32 range");
  return {int(warmup_queries / batch), profile_first_batch ? 1 : int(measurement_queries / batch),
          warmup_queries, profile_first_batch ? batch : measurement_queries};
}

// Fixed storage granularity, independent of execution batches and widths.
constexpr uint64_t kChunkQueries = 4096;

template <typename IndexT>
std::string SampleDescriptor(int rows, int hotness, float alpha,
                             bool shuffle, bool permute) {
  auto descriptor = Descriptor<IndexT>(rows, 0, hotness, alpha, shuffle, permute,
                                      false, std::vector<int>{});
  descriptor.replace(0, std::string("stream-v1").size(), "samples-v2");
  descriptor.erase(descriptor.find(";batch=0"), std::string(";batch=0").size());
  return descriptor + ";chunk_queries=" + std::to_string(kChunkQueries);
}

// generate and consume operate on just one host batch. A hit never constructs
// the generator. Lock lifetime covers readers as well as atomic replacement.
template <typename IndexT, typename Generate, typename Consume>
Result Process(const std::string& directory, const std::string& descriptor,
               uint64_t batches, size_t elements, Generate generate, Consume consume) {
  Require(batches > 0, "batch count must be positive");
  Require(elements <= (std::numeric_limits<uint64_t>::max() - 4) / sizeof(IndexT),
          "batch byte count overflow");
  const uint64_t bytes = elements * sizeof(IndexT);
  Require(bytes <= uint64_t(std::numeric_limits<std::streamsize>::max()),
          "batch exceeds stream I/O limit");
  std::vector<IndexT> host(elements);
  if (directory.empty()) {
    for (uint64_t i = 0; i < batches; ++i) { generate(host); consume(host); }
    return {"disabled", ""};
  }
  std::filesystem::create_directories(directory);
  const std::string path = directory + "/" + Key(descriptor) + ".bin";
  Lock lock(path + ".lock");
  const bool exists = std::filesystem::exists(path);
  const std::string invalid = "invalid cache " + path + "; remove this .bin file and retry";
  if (exists) {
    std::ifstream in(path, std::ios::binary);
    // Four uint64 fields: magic/version, descriptor length, batches, batch bytes.
    Require(Read<uint64_t>(in, path) == 0x4355455354520001ULL, invalid);
    const auto descriptor_size = Read<uint64_t>(in, path);
    const auto stored_batches = Read<uint64_t>(in, path);
    Require(Read<uint64_t>(in, path) == bytes && descriptor_size == descriptor.size(), invalid);
    std::string stored_descriptor(descriptor_size, '\0');
    in.read(stored_descriptor.data(), descriptor_size);
    Require(bool(in) && stored_descriptor == descriptor && stored_batches > 0, invalid);
    const uint64_t header_bytes = 32 + descriptor_size;
    Require(stored_batches <= (std::numeric_limits<uint64_t>::max() - header_bytes) / (bytes + 4), invalid);
    Require(std::filesystem::file_size(path) == header_bytes + stored_batches * (bytes + 4), invalid);
    for (uint64_t i = 0; i < std::min(batches, stored_batches); ++i) {
      const auto expected = Read<uint32_t>(in, path);
      in.read(reinterpret_cast<char*>(host.data()), bytes);
      Require(bool(in) && Crc32c(host.data(), bytes) == expected, invalid + " (checksum)");
      if (stored_batches >= batches) consume(host);
    }
    if (stored_batches >= batches) return {"hit", path};
  }
  TemporaryFile temporary(path);
  std::ofstream out(temporary.path(), std::ios::binary | std::ios::trunc);
  Write(out, uint64_t(0x4355455354520001ULL));
  Write(out, uint64_t(descriptor.size()));
  Write(out, batches);
  Write(out, bytes);
  out.write(descriptor.data(), descriptor.size());
  for (uint64_t i = 0; i < batches; ++i) {
    generate(host);
    Require(host.size() == elements, "generator returned wrong batch size");
    Write(out, Crc32c(host.data(), bytes));
    out.write(reinterpret_cast<const char*>(host.data()), bytes);
    Require(bool(out), "write failed for " + path + "; check scratch space/permissions");
    consume(host);
  }
  out.flush();
  Require(bool(out), "flush failed for " + path);
  out.close();
  Require(!out.fail(), "close failed for " + path);
  const int fd = ::open(temporary.path().c_str(), O_RDONLY);
  Require(fd >= 0, "cannot reopen " + temporary.path());
  const int synced = ::fsync(fd);
  ::close(fd);
  Require(synced == 0, "fsync failed for " + path);
  std::filesystem::rename(temporary.path(), path);
  return {exists ? "replaced" : "created", path};
}

// Consume an exact prefix, regrouping canonical chunks into execution batches.
// The final storage chunk can contain padding samples, never executed samples.
template <typename IndexT, typename Generate, typename Consume>
Result ProcessSamples(const std::string& directory, const std::string& descriptor,
                      uint64_t queries, int batch_size, int hotness,
                      Generate generate_chunk, Consume consume_batch) {
  Require(batch_size > 0 && hotness > 0 && queries > 0 && queries % batch_size == 0,
          "stream query boundary must be exactly divisible by batch size");
  const uint64_t elements = uint64_t(batch_size) * hotness;
  Require(elements <= std::numeric_limits<size_t>::max() / sizeof(IndexT),
          "execution batch size overflow");
  const uint64_t chunks = queries / kChunkQueries + (queries % kChunkQueries != 0);
  std::vector<IndexT> batch;
  batch.reserve(elements);
  uint64_t remaining_queries = queries;
  auto consume_chunk = [&](const std::vector<IndexT>& chunk) {
    const size_t count = std::min(remaining_queries, kChunkQueries) * hotness;
    size_t offset = 0;
    while (offset < count) {
      const size_t take = std::min(size_t(elements) - batch.size(), count - offset);
      batch.insert(batch.end(), chunk.begin() + offset, chunk.begin() + offset + take);
      offset += take;
      if (batch.size() == elements) { consume_batch(batch); batch.clear(); }
    }
    remaining_queries -= count / hotness;
  };
  const auto result = Process<IndexT>(directory, descriptor, chunks,
      kChunkQueries * hotness, generate_chunk, consume_chunk);
  Require(remaining_queries == 0 && batch.empty(), "incomplete stream prefix");
  return result;
}

}  // namespace stream_cache
}  // namespace utils
}  // namespace cuembed
#endif
