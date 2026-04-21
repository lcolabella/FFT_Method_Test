#include <cassert>

#include "core/Grid3D.hpp"

int main() {
    permeability::Grid3D g(4, 5, 6, 1.0, 1.0, 1.0);
    assert(g.total_size() == 120);

    const std::size_t idx = g.flat_index(2, 3, 4);
    auto [i, j, k] = g.unflatten(idx);
    assert(i == 2 && j == 3 && k == 4);

    const std::size_t n = g.periodic_neighbor(0, 0, 0, -1, -1, -1);
    auto [ni, nj, nk] = g.unflatten(n);
    assert(ni == 3 && nj == 4 && nk == 5);

    assert(g.periodic_index(-1, g.nx()) == 3);
    assert(g.periodic_index(5, g.ny()) == 0);

    return 0;
}
