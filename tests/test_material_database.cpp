#include <cassert>
#include <cstdint>
#include <string>

#include "materials/MaterialDatabase.hpp"

int main() {
    common::MaterialDatabase db;

    common::MaterialProperties fluid;
    fluid.values["viscosity"] = 1.0;
    fluid.tags["phase"] = "fluid";
    db.insert(static_cast<std::uint32_t>(1), fluid);

    assert(db.has(1));
    assert(db.hasTag(1, "phase"));
    assert(db.tag(1, "phase") == "fluid");

    // Regression guard: unknown IDs should not throw in hasTag(), only return false.
    assert(!db.has(999));
    assert(!db.hasTag(999, "phase"));

    return 0;
}
