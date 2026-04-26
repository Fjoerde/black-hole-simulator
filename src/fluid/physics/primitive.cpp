#include <cmath>
#include <vector>
#include <array>
#include <stdexcept>
#include "prim.hpp"

// this document includes the method definitions for the primitive
// quantities from the metric components and thermodynamic state.

// compute the lorentz factor
double primitive::lorentz(double v[3], const metriccomp mc) const {
    // compute square norm of 3-velocity
    double v2 = 0.0;
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            v2 += mc.gam[i][j]*v[i]*v[j];
        }
    }
    // error in case velocity squared becomes greater than 1
    if(v2>=1.0) throw std::domain_error("Primitive variable calculations threw back an error: the velocity is superliminal!");
    
    return pow(1.0-v2,-1/2);
}

// compute magnetic quantities
double primitive::bsq(double B[3], double v[3], const metriccomp mc) const {
    double Bsq = 0.0;
    double Bv = 0.0;
    double B_i = 0.0;
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            B_i += mc.gam[i][j]*B[j];
        }
        Bsq += B_i*B[i];
        Bv += B_i*v[i];
    }
    double Y = lorentz(v,mc);
    return (Bsq+Bv*Bv)/(Y*Y);
}

// compute rest of the primitive variables
primvar primitive::comp(double rho, double eps, double v[3], double B[3], double r, double th) const {
    const metriccomp mc = mtr.comp(r,th);
    primvar pv;

    pv.rho = rho;
    pv.eps = eps;
    for(int i=0; i<3; i++) {
        pv.v[i] = v[i];
        pv.B[i] = B[i];
    }

    pv.p = st.press(rho,eps);
    pv.h = st.enth(rho,eps);
    pv.lor = lorentz(v,mc);
    pv.b2 = bsq(B,v,mc);

    return pv;
}