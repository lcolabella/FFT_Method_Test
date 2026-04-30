#include "materials/MaterialDatabase.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <unordered_set>
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
    const auto pos = line.find_first_of("#;");
    if (pos != std::string::npos) {
        line = line.substr(0, pos);
    }
    return trim(line);
}

bool isIdentifierChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.';
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

MaterialDatabase MaterialDatabase::read(const std::string& filePath) {
    std::ifstream input(filePath);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open material file: " + filePath);
    }

    MaterialDatabase db;
    std::string line;
    std::size_t lineNo = 0;

    while (std::getline(input, line)) {
        ++lineNo;
        line = stripComment(line);
        if (line.empty()) {
            continue;
        }

        std::istringstream stream(line);
        std::uint32_t materialId = 0;
        if (!(stream >> materialId)) {
            throw std::runtime_error(
                "Invalid material line " + std::to_string(lineNo) + ": expected leading integer ID");
        }

        MaterialProperties props;
        std::string token;
        while (stream >> token) {
            const auto eqPos = token.find('=');
            if (eqPos == std::string::npos || eqPos == 0 || eqPos == token.size() - 1) {
                throw std::runtime_error(
                    "Invalid material property token at line " + std::to_string(lineNo) + ": " + token);
            }

            const std::string key = token.substr(0, eqPos);
            const std::string valueString = token.substr(eqPos + 1);
            if (!std::all_of(key.begin(), key.end(), isIdentifierChar)) {
                throw std::runtime_error(
                    "Invalid material property name at line " + std::to_string(lineNo) + ": " + key);
            }

            const std::string loweredKey = toLower(key);
            if (loweredKey == "phase") {
                const std::string loweredValue = toLower(valueString);
                if (loweredValue != "fluid" && loweredValue != "solid") {
                    throw std::runtime_error(
                        "Invalid phase value at line " + std::to_string(lineNo) +
                        ": expected fluid|solid");
                }
                props.tags[loweredKey] = loweredValue;
                continue;
            }

            try {
                std::size_t used = 0;
                const double value = std::stod(valueString, &used);
                if (used != valueString.size()) {
                    throw std::runtime_error("");
                }
                props.values[loweredKey] = value;
            } catch (...) {
                throw std::runtime_error(
                    "Invalid numeric value for property '" + key + "' at line " +
                    std::to_string(lineNo));
            }
        }

        if (props.values.empty() && props.tags.empty()) {
            throw std::runtime_error(
                "Material line " + std::to_string(lineNo) + " has no properties");
        }

        db.materials_[materialId] = std::move(props);
    }

    if (db.materials_.empty()) {
        throw std::runtime_error("Material file has no material definitions: " + filePath);
    }

    return db;
}

void MaterialDatabase::insert(std::uint32_t materialId, MaterialProperties props) {
    materials_[materialId] = std::move(props);
}

bool MaterialDatabase::has(std::uint32_t materialId) const {
    return materials_.find(materialId) != materials_.end();
}

const MaterialProperties& MaterialDatabase::at(std::uint32_t materialId) const {
    const auto it = materials_.find(materialId);
    if (it == materials_.end()) {
        throw std::runtime_error("Missing properties for material id: " + std::to_string(materialId));
    }
    return it->second;
}

bool MaterialDatabase::hasTag(std::uint32_t materialId, const std::string& key) const {
    if (!has(materialId)) {
        return false;
    }
    const auto& props = at(materialId);
    return props.tags.find(toLower(key)) != props.tags.end();
}

std::string MaterialDatabase::tag(std::uint32_t materialId, const std::string& key) const {
    const auto& props = at(materialId);
    const auto it = props.tags.find(toLower(key));
    if (it == props.tags.end()) {
        throw std::runtime_error(
            "Missing tag '" + key + "' for material id: " + std::to_string(materialId));
    }
    return it->second;
}

std::vector<std::uint32_t> MaterialDatabase::findMissing(const Geometry& geometry) const {
    std::vector<std::uint32_t> missing;
    std::unordered_set<std::uint32_t> seen;

    for (std::uint32_t materialId : geometry.materialIds) {
        if (materials_.find(materialId) != materials_.end()) {
            continue;
        }
        if (seen.insert(materialId).second) {
            missing.push_back(materialId);
        }
    }

    std::sort(missing.begin(), missing.end());
    return missing;
}

std::vector<std::uint32_t> MaterialDatabase::idsForTag(const std::string& key, const std::string& value) const {
    const std::string loweredKey = toLower(key);
    const std::string loweredValue = toLower(value);

    std::vector<std::uint32_t> ids;
    for (const auto& [materialId, props] : materials_) {
        const auto it = props.tags.find(loweredKey);
        if (it != props.tags.end() && toLower(it->second) == loweredValue) {
            ids.push_back(materialId);
        }
    }

    std::sort(ids.begin(), ids.end());
    return ids;
}

std::size_t MaterialDatabase::size() const {
    return materials_.size();
}

} // namespace common
