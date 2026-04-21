#include "io/VoxelIO.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace permeability {

BinaryMedium VoxelIO::load_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("VoxelIO::load_text cannot open file: " + path);
    }

    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;
    if (!(in >> nx >> ny >> nz)) {
        throw std::runtime_error("VoxelIO::load_text expected first line: nx ny nz");
    }

    Grid3D grid(nx, ny, nz);
    std::vector<unsigned char> mask(grid.total_size(), 0);

    for (std::size_t idx = 0; idx < grid.total_size(); ++idx) {
        int v = 0;
        if (!(in >> v)) {
            throw std::runtime_error("VoxelIO::load_text insufficient voxel values");
        }
        if (v != 0 && v != 1) {
            throw std::runtime_error("VoxelIO::load_text voxel values must be 0 or 1");
        }
        mask[idx] = static_cast<unsigned char>(v);
    }

    return BinaryMedium(grid, std::move(mask));
}

}  // namespace permeability
