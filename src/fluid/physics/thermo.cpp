#include "state.hpp"
#include <cmath>
#include <stdexcept>

// this document contains the methods for calculating thermodynamic
// properties of the gas. diagnostic tests throughout are included
// to ensure quantities remain physical

// compute pressure
double state::press(double rho, double eps) const {
    if(rho<=0.0) throw std::domain_error("Thermodynamic calculations threw back an error: the density is not positive!");
    return (gamma-1.0)*rho*eps;
}
// compute specific enthalpy
double state::enth(double rho, double eps) const {
    if(rho<=0.0) throw std::domain_error("Thermodynamic calculations threw back an error: the density is not positive!");
    return 1.0+gamma*eps;
}
// compute internal energy
double state::eng(double rho, double p) const {
    if(rho<=0.0) throw std::domain_error("Thermodynamic calculations threw back an error: the density is not positive!");
    return p/((gamma-1.0)*rho);
}
// compute temperature
double state::temp(double rho, double p) const {
    if(rho<=0.0) throw std::domain_error("Thermodynamic calculations threw back an error: the density is not positive!");
    return p/rho;
}
// compute speed of sound squared
double state::cs2(double rho, double eps) const{
    if(rho<=0.0) throw std::domain_error("Thermodynamic calculations threw back an error: the density is not positive!");
    double p = press(rho, eps);
    double h = enth(rho, eps);
    return (gamma*p)/(rho*h);
}