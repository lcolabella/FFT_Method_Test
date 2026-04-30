#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "io/VoxelIO.hpp"

namespace {

void write_fgeo_u16(const std::string& path,
                    std::uint64_t nx,
                    std::uint64_t ny,
                    std::uint64_t nz,
                    const std::vector<std::uint16_t>& values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    assert(out.is_open());

    const char magic[4] = {'F', 'G', 'E', 'O'};
    const std::uint8_t version = 1;
    const std::uint8_t value_type = 2;  // uint16 payload
    const std::uint16_t reserved = 0;

    out.write(magic, 4);
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&value_type), sizeof(value_type));
    out.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    out.write(reinterpret_cast<const char*>(&nx), sizeof(nx));
    out.write(reinterpret_cast<const char*>(&ny), sizeof(ny));
    out.write(reinterpret_cast<const char*>(&nz), sizeof(nz));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(std::uint16_t)));
}

}  // namespace

int main() {
    const std::string text_path = "/tmp/fft_method_step8_geometry.geom";
    const std::string fgeo_path = "/tmp/fft_method_step8_geometry.fgeo";

    {
        std::ofstream out(text_path, std::ios::trunc);
        assert(out.is_open());
        out << "# comment\n";
        out << "2 2 1\n";
        out << "0 5\n";
        out << "9 0\n";
    }

    const common::Geometry text_geo = permeability::VoxelIO::load_geometry(text_path);
    assert(text_geo.nx == 2);
    assert(text_geo.ny == 2);
    assert(text_geo.nz == 1);
    assert(text_geo.materialIds.size() == 4);
    assert(text_geo.materialIds[0] == 0);
    assert(text_geo.materialIds[1] == 5);
    assert(text_geo.materialIds[2] == 9);
    assert(text_geo.materialIds[3] == 0);

    write_fgeo_u16(fgeo_path, 2, 2, 1, {1, 7, 9, 1});
    const common::Geometry fgeo_geo = permeability::VoxelIO::load_geometry(fgeo_path);
    assert(fgeo_geo.nx == 2);
    assert(fgeo_geo.ny == 2);
    assert(fgeo_geo.nz == 1);
    assert(fgeo_geo.materialIds.size() == 4);
    assert(fgeo_geo.materialIds[0] == 1);
    assert(fgeo_geo.materialIds[1] == 7);
    assert(fgeo_geo.materialIds[2] == 9);
    assert(fgeo_geo.materialIds[3] == 1);

    std::remove(text_path.c_str());
    std::remove(fgeo_path.c_str());
    return 0;
}
