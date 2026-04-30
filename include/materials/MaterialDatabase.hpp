#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/Geometry.hpp"

namespace common {

struct MaterialProperties {
    std::unordered_map<std::string, double> values;
    std::unordered_map<std::string, std::string> tags;
};

class MaterialDatabase {
public:
    static MaterialDatabase read(const std::string& filePath);

    // Programmatic insertion; used when building a DB without a file.
    void insert(std::uint32_t materialId, MaterialProperties props);

    bool has(std::uint32_t materialId) const;
    const MaterialProperties& at(std::uint32_t materialId) const;
    bool hasTag(std::uint32_t materialId, const std::string& key) const;
    std::string tag(std::uint32_t materialId, const std::string& key) const;
    std::vector<std::uint32_t> findMissing(const Geometry& geometry) const;
    std::vector<std::uint32_t> idsForTag(const std::string& key, const std::string& value) const;
    std::size_t size() const;

private:
    std::unordered_map<std::uint32_t, MaterialProperties> materials_;
};

} // namespace common

