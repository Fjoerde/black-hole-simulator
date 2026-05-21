#pragma once
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include "metric.hpp"
#include "grid.hpp"
#include "ct.hpp"

// this document contains the structures and classes for variable
// initialisation and the prescribing of initial conditions.
// fishbone-moncrief torus initialisation
using namespace grid;

namespace torus {
    // magnetic initialisation
class init {
public:
    static void fm_init(amrtree& tree);
    static void B_pot_init(patch& p, const metric& mtr, double glmx_rho);
private:
    static std::array<double,3> A_vpot(double x, double y, double z, double rho, double rho_max);
};
}