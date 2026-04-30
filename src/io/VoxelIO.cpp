#include "io/VoxelIO.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace permeability {

namespace {

// Reads a little-endian value of type T from raw bytes at offset.
template <typename T>
T read_le(const unsigned char* data) {
    T value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        value |= static_cast<T>(data[i]) << (8 * i);
    }
    return value;
}

bool ends_with_fgeo(const std::string& path) {
    const std::string ext = ".fgeo";
    return path.size() >= ext.size() &&
           path.compare(path.size() - ext.size(), ext.size(), ext) == 0;
}

} // namespace

// Skip lines starting with # or ; (comments) and return the next non-empty token stream.
static std::istringstream next_content_line(std::istream& in) {
    std::string line;
    while (std::getline(in, line)) {
        // Strip inline comments
        const auto comment = line.find_first_of("#;");
        if (comment != std::string::npos) line = line.substr(0, comment);
        // Trim whitespace
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        return std::istringstream(line.substr(first));
    }
    return std::istringstream{};
}

BinaryMedium VoxelIO::load_text(const std::string& path, Scalar voxel_size, int fluid_id) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("VoxelIO::load_text cannot open file: " + path);
    }

    // Read dimensions from first non-comment line
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;
    {
        auto ss = next_content_line(in);
        if (!(ss >> nx >> ny >> nz)) {
            throw std::runtime_error("VoxelIO::load_text expected first non-comment line: nx ny nz");
        }
    }

    Grid3D grid(nx, ny, nz, voxel_size, voxel_size, voxel_size);
    std::vector<unsigned char> mask;
    mask.reserve(grid.total_size());

    // Read remaining tokens, skipping comment lines, until we have all voxels
    std::string line;
    while (mask.size() < grid.total_size() && std::getline(in, line)) {
        const auto comment = line.find_first_of("#;");
        if (comment != std::string::npos) line = line.substr(0, comment);
        std::istringstream ss(line);
        int v = 0;
        while (ss >> v) {
            mask.push_back(static_cast<unsigned char>(v == fluid_id ? 1 : 0));
            if (mask.size() == grid.total_size()) break;
        }
    }

    if (mask.size() < grid.total_size()) {
        throw std::runtime_error("VoxelIO::load_text insufficient voxel values");
    }

    return BinaryMedium(grid, std::move(mask));
}

BinaryMedium VoxelIO::load_fgeo(const std::string& path, Scalar voxel_size, int fluid_id) {
    // Read entire file
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("VoxelIO::load_fgeo cannot open file: " + path);
    }
    const std::vector<unsigned char> data(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());

    // Header layout (little-endian):
    //   offset 0 : 4 bytes  magic "FGEO"
    //   offset 4 : 1 byte   version (must be 1)
    //   offset 5 : 1 byte   value_type (1=uint32, 2=uint16)
    //   offset 6 : 2 bytes  reserved
    //   offset 8 : 8 bytes  nx (uint64)
    //   offset 16: 8 bytes  ny (uint64)
    //   offset 24: 8 bytes  nz (uint64)
    //   offset 32: payload
    constexpr std::size_t HEADER_SIZE = 32;
    if (data.size() < HEADER_SIZE) {
        throw std::runtime_error("VoxelIO::load_fgeo file too small: " + path);
    }

    if (data[0] != 'F' || data[1] != 'G' || data[2] != 'E' || data[3] != 'O') {
        throw std::runtime_error("VoxelIO::load_fgeo invalid magic in: " + path);
    }
    if (data[4] != 1) {
        throw std::runtime_error("VoxelIO::load_fgeo unsupported version " +
                                 std::to_string(data[4]) + " in: " + path);
    }
    const std::uint8_t value_type = data[5];
    if (value_type != 1 && value_type != 2) {
        throw std::runtime_error("VoxelIO::load_fgeo unsupported value_type " +
                                 std::to_string(value_type) + " in: " + path);
    }

    const std::uint64_t nx = read_le<std::uint64_t>(data.data() + 8);
    const std::uint64_t ny = read_le<std::uint64_t>(data.data() + 16);
    const std::uint64_t nz = read_le<std::uint64_t>(data.data() + 24);
    const std::uint64_t count = nx * ny * nz;

    const std::size_t elem_bytes = (value_type == 1) ? 4 : 2;
    const std::size_t expected = HEADER_SIZE + count * elem_bytes;
    if (data.size() != expected) {
        throw std::runtime_error(
            "VoxelIO::load_fgeo payload size mismatch in: " + path +
            " (expected " + std::to_string(expected) +
            ", got " + std::to_string(data.size()) + ")");
    }

    Grid3D grid(static_cast<std::size_t>(nx),
                static_cast<std::size_t>(ny),
                static_cast<std::size_t>(nz),
                voxel_size, voxel_size, voxel_size);
    std::vector<unsigned char> mask;
    mask.reserve(static_cast<std::size_t>(count));

    const unsigned char* payload = data.data() + HEADER_SIZE;
    for (std::uint64_t i = 0; i < count; ++i) {
        int mat_id = 0;
        if (value_type == 1) {
            mat_id = static_cast<int>(read_le<std::uint32_t>(payload + i * 4));
        } else {
            mat_id = static_cast<int>(read_le<std::uint16_t>(payload + i * 2));
        }
        mask.push_back(mat_id == fluid_id ? 1 : 0);
    }

    return BinaryMedium(grid, std::move(mask));
}

BinaryMedium VoxelIO::load(const std::string& path, Scalar voxel_size, int fluid_id) {
    if (ends_with_fgeo(path)) {
        return load_fgeo(path, voxel_size, fluid_id);
    }
    return load_text(path, voxel_size, fluid_id);
}

common::Geometry VoxelIO::load_geometry(const std::string& path) {
    if (ends_with_fgeo(path)) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("VoxelIO::load_geometry cannot open file: " + path);
        }
        const std::vector<unsigned char> data(
            (std::istreambuf_iterator<char>(in)),
            std::istreambuf_iterator<char>());

        constexpr std::size_t HEADER_SIZE = 32;
        if (data.size() < HEADER_SIZE) {
            throw std::runtime_error("VoxelIO::load_geometry fgeo too small: " + path);
        }
        if (data[0] != 'F' || data[1] != 'G' || data[2] != 'E' || data[3] != 'O') {
            throw std::runtime_error("VoxelIO::load_geometry invalid magic: " + path);
        }
        if (data[4] != 1) {
            throw std::runtime_error("VoxelIO::load_geometry unsupported version: " + path);
        }
        const std::uint8_t value_type = data[5];
        if (value_type != 1 && value_type != 2) {
            throw std::runtime_error("VoxelIO::load_geometry unsupported value_type: " + path);
        }
        const std::uint64_t nx = read_le<std::uint64_t>(data.data() + 8);
        const std::uint64_t ny = read_le<std::uint64_t>(data.data() + 16);
        const std::uint64_t nz = read_le<std::uint64_t>(data.data() + 24);
        const std::uint64_t count = nx * ny * nz;
        const std::size_t elem_bytes = (value_type == 1) ? 4U : 2U;
        if (data.size() != HEADER_SIZE + count * elem_bytes) {
            throw std::runtime_error("VoxelIO::load_geometry fgeo payload mismatch: " + path);
        }
        common::Geometry geo;
        geo.nx = static_cast<std::size_t>(nx);
        geo.ny = static_cast<std::size_t>(ny);
        geo.nz = static_cast<std::size_t>(nz);
        geo.materialIds.reserve(static_cast<std::size_t>(count));
        const unsigned char* payload = data.data() + HEADER_SIZE;
        for (std::uint64_t i = 0; i < count; ++i) {
            if (value_type == 1) {
                geo.materialIds.push_back(read_le<std::uint32_t>(payload + i * 4));
            } else {
                geo.materialIds.push_back(
                    static_cast<std::uint32_t>(read_le<std::uint16_t>(payload + i * 2)));
            }
        }
        return geo;
    }

    // Text format
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("VoxelIO::load_geometry cannot open file: " + path);
    }
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;
    {
        auto ss = next_content_line(in);
        if (!(ss >> nx >> ny >> nz)) {
            throw std::runtime_error(
                "VoxelIO::load_geometry expected first non-comment line: nx ny nz");
        }
    }
    common::Geometry geo;
    geo.nx = nx;
    geo.ny = ny;
    geo.nz = nz;
    geo.materialIds.reserve(nx * ny * nz);
    std::string line;
    while (geo.materialIds.size() < nx * ny * nz && std::getline(in, line)) {
        const auto comment = line.find_first_of("#;");
        if (comment != std::string::npos) line = line.substr(0, comment);
        std::istringstream ss(line);
        unsigned int v = 0;
        while (ss >> v) {
            geo.materialIds.push_back(static_cast<std::uint32_t>(v));
            if (geo.materialIds.size() == nx * ny * nz) break;
        }
    }
    if (geo.materialIds.size() < nx * ny * nz) {
        throw std::runtime_error("VoxelIO::load_geometry insufficient voxel values");
    }
    return geo;
}

}  // namespace permeability
