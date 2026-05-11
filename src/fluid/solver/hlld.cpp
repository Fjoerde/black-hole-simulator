#include <cmath>
#include <vector>
#include <array>

#include "hlld.hpp"

// this document contains the methods for the HLLD solver.

// define magnetic 4-vector
std::array<double,4> hlld::Bfl(const prim& W, const metriccomp& mc) {
    std::array<double,4> b = {0.0, 0.0, 0.0, 0.0};
    double vB = 0.0;
    double v_i = 0.0;
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            v_i += mc.gam[i][j]*W.v[j];
        }
        vB += v_i*W.B[i];
    }
    b[0] += W.lor*vB/mc.alpha;
    for(int i=0; i<3; i++) {
        b[i+1]=(W.B[i]+mc.alpha*b[0]*W.lor*W.v[i])/W.lor;
    }
    return b;
}