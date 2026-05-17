#pragma once
#include <cmath>
#include <vector>
#include <array>
#include "grid.hpp"
#include "cell.hpp"
#include "metric.hpp"
#include "hlld.hpp"

// this document contains the structures and declarations for methods
// relating to the second-order runge-kutta integrator used to
// advance the valencia evolution equation.

class rk2integrator {
public:
    // courant-friedrichs-lewy number
    double cfl;
    explicit rk2integrator(double cfl_in = 0.5) : cfl(cfl_in) {}
    // time step
    double step(amrtree& tree);
    static cons div_flux(const patch& p, int i, int j, int k);
    static cons source(const patch& p, int i, int j, int k, const metric& mtr, const state& stt);
    void rkstg(amrtree& tree, double dt, int stg);
    static double dtcomp(const amrtree& tree, double cfl);
};