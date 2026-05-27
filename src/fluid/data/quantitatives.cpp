#include <cmath>
#include <vector>
#include <array>
#include <functional>
#include "../grid.hpp"
#include "../cell.hpp"
#include "../analysis.hpp"

// this document contains the methods relating to quantitative
// analysis of the black hole and accretion disc.

using namespace grid;

// integrate over a volume
double analysis::vol_int(const grid::amrtree& tree, std::function<double(const cell&, double)> f) {
    
}