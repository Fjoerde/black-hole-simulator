#pragma once
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include "prim.hpp"
#include "cell.hpp"
#include "grid.hpp"

// this document contains the declarations for the reconstruction of
// left and right states at each face using data at cell centres.

// slope limiting functions
// minimum modulus for two and three variables
inline double minmod(double a, double b) {
    if(a*b<=0.0) return 0.0;
    return (std::abs(a)<std::abs(b))? a : b;
}
inline double min3mod(double a, double b, double c) {
    return minmod(a, minmod(b,c));
}
// van leer limiter
inline double vanleer(double a, double b) {
    if(a*b<=0.0) return 0.0;
    return 2.0*a*b/(a+b);
}
// monotonised central limiter
inline double mclim(double a, double b) {
    if(a*b<=0.0) return 0.0;
    double c = (a+b)/2;
    double s = (a>0.0)? 1.0 : -1.0;
    return s*min3mod(std::abs(2*a),std::abs(2*b),std::abs(c));
}

// face state reconstruction
struct facestate {
    prim L; prim R;
};
enum class limiter {minmod, vanleer, mclim};
facestate reconfp(const patch& p, int i, int j, int k, int dim, limiter lim = limiter::vanleer);