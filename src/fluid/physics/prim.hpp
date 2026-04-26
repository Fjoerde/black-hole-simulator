#pragma once
#include <cmath>
#include <vector>
#include <array>
#include <iostream>

#include "metric.hpp"
#include "state.hpp"

// this document contains the structures for computing the primitive
// variable vector \mathbf{W}. several important primitive quantities
// are derived from the metric and equation of state.

struct primvar {
    // primitive variables
    double rho; // rest mass density
    double eps; // specific internal energy
    double v[3]; // 3-velocity components
    double B[3]; // magnetic field components

    // derived quantities
    double p; // pressure
    double h; // enthalpy
    double lor; // Lorentz factor
    double b2; // double the magnetic field energy density b^{\mu}b_{\mu}
};

class primitive {
public:
    const metric& mtr;
    const state& st;

    primitive(const metric& m, const state& s) : mtr(m), st(s) {};

    // compute quantities
    primvar comp(double rho, double eps, double v[3], double B[3], double r, double th) const;
    double lorentz(double v[3], const metriccomp mc) const;
    double bsq(double B[3], double v[3], const metriccomp mc) const;    
};