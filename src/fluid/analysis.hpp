#pragma once
#include <cmath>
#include <vector>
#include <array>
#include <functional>
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

class analysis {
public:
    double vol_int(const grid::amrtree& tree, const std::function<double(const cell&, double)> f);
    double surf_int(const grid::amrtree tree, const double r_surf, std::function<double(const cell&, double)> f);
    bundle results();
};
