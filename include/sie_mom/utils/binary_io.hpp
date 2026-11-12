#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sie_mom {
namespace utils {

/**
 * @brief Ensures host system is little-endian for portable binary serialization.
 */
inline void require_little_endian_host() {
    const std::uint16_t endian_probe = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian_probe) != 1) {
        throw std::runtime_error("binary file I/O requires a little-endian host");
    }
}

/**
 * @brief Writes a contiguous vector of plain-old-data (POD) values to a binary file.
 * Creates parent directories automatically if they do not exist.
 *
 * @tparam T Trivially copyable element type (e.g. real_t, double, float, complex__float)
 * @param path Destination filesystem path
 * @param data Contiguous vector to write
 */
template <typename T>
inline void write_binary_vector(const std::filesystem::path& path, const std::vector<T>& data) {
    require_little_endian_host();
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open binary output file for writing: " + path.string());
    }
    out.write(
        reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(T))
    );
    if (!out) {
        throw std::runtime_error("failed while writing binary output file: " + path.string());
    }
}

/**
 * @brief Reads a contiguous vector of plain-old-data (POD) values from a binary file.
 * Validates that the file exists and its byte size matches data.size() * sizeof(T).
 *
 * @tparam T Trivially copyable element type
 * @param path Source filesystem path
 * @param data Pre-allocated vector to read into
 * @return true if successfully loaded and verified, false if file missing or size mismatch
 */
template <typename T>
inline bool read_binary_vector(const std::filesystem::path& path, std::vector<T>& data) {
    require_little_endian_host();
    if (!std::filesystem::exists(path)) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const auto file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (file_size != static_cast<std::streampos>(data.size() * sizeof(T))) {
        return false;
    }
    in.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(T))
    );
    return in.good();
}

/**
 * @brief Reads a vector from a binary file, determining size automatically from file length.
 * @tparam T Trivially copyable element type
 * @param path Source filesystem path
 * @return std::vector<T> loaded from file
 */
template <typename T>
inline std::vector<T> read_binary_vector(const std::filesystem::path& path) {
    require_little_endian_host();
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Binary file not found: " + path.string());
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open binary file: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const auto file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (file_size % sizeof(T) != 0) {
        throw std::runtime_error("File size " + std::to_string(file_size) + " is not a multiple of " + std::to_string(sizeof(T)));
    }
    std::vector<T> data(file_size / sizeof(T));
    in.read(
        reinterpret_cast<char*>(data.data()),
        file_size
    );
    if (!in) {
        throw std::runtime_error("Failed reading binary data from " + path.string());
    }
    return data;
}

} // namespace utils
} // namespace sie_mom

