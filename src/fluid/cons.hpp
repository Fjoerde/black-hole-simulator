#pragma once
#include <cmath>
#include <vector>
#include <array>
#include <iostream>

#include "prim.hpp"
#include "state.hpp"
#include "metric.hpp"

// this document contains the structures for conserved variables
// and their construction from the primitive variables.

struct cons {
    double D; // conserved particle density
    double S[3]; // conserved momentum
    double tau; // conserved energy
    double B[3]; // conserved magnetic flux
};

class conserved {
public:
    // required parameters
    const metric& mtr;
    const state& stt;
    const primitive& prm;
    
    conserved(const metric& m, const state& s, const primitive& p) : mtr(m), stt(s), prm(p) {}

    // primitive to conserved construction
    cons ptoc(prim& pv, double r, double th) const;

    // conserved to primitive reconstruction
    // false returned when root solver fails and diverges
    bool ctop(const cons& cv, double r, double th, prim& pv_out, int maxiter=50, double tol=1e-8) const;
    bool cp_bd(double xa, double xb, double Ssq, double Bsq, double SB, double D, double tau, double& root, double tol=1e-12) const;

private:
    // residual and derivative
    double f_df(double xi, double Ssq, double Bsq, double SB, double Dn, double taun, double& df) const;
    double fres(double xi, double Ssq, double Bsq, double SB, double Dn, double taun) const;
};
// minmod limiting
inline double minmod(double a, double b) {
    if(a*b<=0) {return 0.0;}
    if(std::abs(a)<std::abs(b)) return a;
    return b;
}
inline double min3mod(double a, double b, double c) {
    return minmod(a,minmod(b,c));
}