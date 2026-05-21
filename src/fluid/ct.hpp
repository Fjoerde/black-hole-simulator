#pragma once
#include <cmath>
#include <vector>
#include <array>
#include "metric.hpp"
#include "grid.hpp"

// this document contains the structures pertaining to constrained
// transport of the magnetic field maintaining \div B = 0.

using namespace grid;

class constrans {
public:
    static void emfcomp(patch& p);
    static void Bfupdate(patch& p, double dt);
    static void f2cB(patch& p);
    static double maxdivB(const patch& p);
};