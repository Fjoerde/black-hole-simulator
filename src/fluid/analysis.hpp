#pragma once
#include <cmath>
#include <vector>
#include <array>
#include "cell.hpp"
#include "grid.hpp"

// this document contains the structures and classes relating to the
// quantitative analysis of the gas.
using namespace grid;

// quantitative results bundle
struct bundle {
    // general quantities
    double bright; // brightness (intensity)
    double lum; // luminosity of the disc
    double energy; // total energy
    double mass; // total mass
    // plasmatic quantities
};

// class analysis {
// public:
//     double lumin(amrtree);
// };