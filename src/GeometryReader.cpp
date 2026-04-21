#include "common/GeometryReader.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <ios>
#include <sstream>
#include <stdexcept>

namespace common {

namespace {

std::string trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string stripComment(std::string line) {
    const auto pos = line.find('#');
    if (pos != std::string::npos) {
        line = line.substr(0, pos);
    }
    return trim(line);
}

bool hasSuffix(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin(), [](char a, char b) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(a))) ==
               static_cast<char>(std::tolower(static_cast<unsigned char>(b)));
    });
}

GeometryFormat detectFormat(const std::string& filePath) {
    if (hasSuffix(filePath, ".fgeo") || hasSuffix(filePath, ".bin")) {
        return GeometryFormat::Binary;
    }
    return GeometryFormat::Text;
}

Geometry readTextGeometry(const std::string& filePath) {
    std::ifstream input(filePath);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open geometry file: " + filePath);
    }

    Geometry geometry;
    std::string line;
    bool headerRead = false;

    while (std::getline(input, line)) {
        line = stripComment(line);
        if (line.empty()) {
            continue;
        }

        std::istringstream headerStream(line);
        if (headerStream >> geometry.nx >> geometry.ny >> geometry.nz) {
            headerRead = true;
            break;
        }
        throw std::runtime_error("Invalid geometry header in file: " + filePath);
    }

    if (!headerRead) {
        throw std::runtime_error("Invalid geometry header in file: " + filePath);
    }

    if (geometry.nx == 0 || geometry.ny == 0 || geometry.nz == 0) {
        throw std::runtime_error("Geometry dimensions must be > 0");
    }

    const std::size_t expected = geometry.voxelCount();
    geometry.materialIds.reserve(expected);

    while (std::getline(input, line)) {
        line = stripComment(line);
        if (line.empty()) {
            continue;
        }

        std::istringstream dataStream(line);
        std::uint32_t materialId = 0;
        while (dataStream >> materialId) {
            geometry.materialIds.push_back(materialId);
        }
    }

    if (geometry.materialIds.size() != expected) {
        throw std::runtime_error(
            "Geometry value count mismatch. Expected " + std::to_string(expected) +
            " but got " + std::to_string(geometry.materialIds.size()));
    }

    return geometry;
}

Geometry readBinaryGeometry(const std::string& filePath) {
    std::ifstream input(filePath, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open geometry file: " + filePath);
    }

    std::array<char, 4> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || magic[0] != 'F' || magic[1] != 'G' || magic[2] != 'E' || magic[3] != 'O') {
        throw std::runtime_error("Invalid binary geometry magic header in: " + filePath);
    }

    std::uint8_t version = 0;
    input.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!input || version != 1) {
        throw std::runtime_error("Unsupported binary geometry version in: " + filePath);
    }

    std::uint8_t valueType = 0;
    input.read(reinterpret_cast<char*>(&valueType), sizeof(valueType));
    if (!input || (valueType != 1 && valueType != 2)) {
        throw std::runtime_error("Unsupported binary geometry value type in: " + filePath);
    }

    std::uint16_t reserved = 0;
    input.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
    if (!input) {
        throw std::runtime_error("Failed reading binary geometry header in: " + filePath);
    }

    std::uint64_t nx = 0;
    std::uint64_t ny = 0;
    std::uint64_t nz = 0;
    input.read(reinterpret_cast<char*>(&nx), sizeof(nx));
    input.read(reinterpret_cast<char*>(&ny), sizeof(ny));
    input.read(reinterpret_cast<char*>(&nz), sizeof(nz));
    if (!input || nx == 0 || ny == 0 || nz == 0) {
        throw std::runtime_error("Invalid binary geometry dimensions in: " + filePath);
    }

    Geometry geometry;
    geometry.nx = static_cast<std::size_t>(nx);
    geometry.ny = static_cast<std::size_t>(ny);
    geometry.nz = static_cast<std::size_t>(nz);

    const std::size_t expected = geometry.voxelCount();
    geometry.materialIds.resize(expected);

    if (valueType == 1) {
        input.read(
            reinterpret_cast<char*>(geometry.materialIds.data()),
            static_cast<std::streamsize>(expected * sizeof(std::uint32_t)));
        if (!input) {
            throw std::runtime_error("Binary geometry payload is truncated in: " + filePath);
        }
    } else {
        std::vector<std::uint16_t> packed(expected, 0);
        input.read(
            reinterpret_cast<char*>(packed.data()),
            static_cast<std::streamsize>(expected * sizeof(std::uint16_t)));
        if (!input) {
            throw std::runtime_error("Binary geometry payload is truncated in: " + filePath);
        }

        for (std::size_t i = 0; i < expected; ++i) {
            geometry.materialIds[i] = static_cast<std::uint32_t>(packed[i]);
        }
    }

    return geometry;
}

} // namespace

Geometry GeometryReader::read(const std::string& filePath, GeometryFormat format) {
    const GeometryFormat effective = (format == GeometryFormat::Auto) ? detectFormat(filePath) : format;
    if (effective == GeometryFormat::Binary) {
        return readBinaryGeometry(filePath);
    }
    return readTextGeometry(filePath);
}

} // namespace common
