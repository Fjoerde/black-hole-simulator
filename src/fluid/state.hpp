#pragma once
#include <cmath>
#include <vector>
#include <array>

// this document holds the class for the equation of state.
// we use an adiabatic ideal gas.

class state {
public:
    // adiabatic gas stuff
    double gamma; // adiabatic index
    explicit state(double gm = 5.0/3.0) : gamma(gm) {};

    // thermodynamic parameters
    double press(double rho, double eps) const; // pressure
    double enth(double rho, double eps) const; // specific enthalpy
    double eng(double rho, double p) const; // internal energy
    double temp(double rho, double p) const; // temperature
    double cs2(double rho, double eps) const; // speed of sound squared
};