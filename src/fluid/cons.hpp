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
    // constructor
    cons() : D(0.0), tau(0.0) {
        S[0] = S[1] = S[2] = 0.0;
        B[0] = B[1] = B[2] = 0.0;
    }

    // operators
    cons operator+(const cons& k) const {
        cons r;
        r.D = D + k.D; r.tau = tau + k.tau;
        for(int i=0; i<3; i++) {
            r.S[i] = S[i] + k.S[i];
            r.B[i] = B[i] + k.B[i];
        }
        return r;
    }
    cons operator-(const cons& k) const {
        cons r;
        r.D = D - k.D; r.tau = tau - k.tau;
        for(int i=0; i<3; i++) {
            r.S[i] = S[i] - k.S[i];
            r.B[i] = B[i] - k.B[i];
        }
        return r;
    }
    cons operator*(double s) const {
        cons r;
        r.D = D*s; r.tau = tau*s;
        for(int i=0; i<3; i++) {
            r.S[i] = S[i]*s; r.B[i] = B[i]*s;
        }
        return r;
    }
    cons& operator+=(const cons& k) {
        D += k.D; tau += k.tau;
        for(int i=0; i<3; i++) {
            S[i] += k.S[i]; B[i] += k.B[i];
        }
        return *this;
    }
};
inline cons operator*(double s, const cons& k) {return k*s;}

class conserved {
public:
    // required parameters
    const metric& mtr;
    const state& stt;
    
    conserved(const metric& m, const state& s) : mtr(m), stt(s) {}

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