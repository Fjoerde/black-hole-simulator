#pragma once
#include <cmath>
#include <vector>
#include <array>

#include "metric.hpp"
#include "grid.hpp"
#include "cell.hpp"
#include "prim.hpp"
#include "cons.hpp"

// this document contains the structures for the riemann hlld solver.

// flux struct
struct hlldflux {
    cons F;
    double ctspeed;
};
hlldflux hlld(const prim& WL, const prim& WR, const cons& UL, const cons& UR, const metric& mtr, double r, double th, const state& stt, int dim);