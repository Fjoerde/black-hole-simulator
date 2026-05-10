#pragma once
#include <cmath>
#include <vector>
#include <array>

#include "metric.hpp"
#include "grid.hpp"
#include "cell.hpp"
#include "prim.hpp"
#include "cons.hpp"

// this document contains the structures and classes for the riemann
// hlld solver.

class hlld {
public:
    struct wave {
        double S_L; double S_R; // magnetosonic waves
        double Sx_L; double Sx_R; // alfvén waves
        double Sx_ct; // contact wave S^*
    };
    // flux, with dir = 0,1,2 to choose x,y,z flux
    static cons flux(const prim& W_L, const prim& W_R, const metriccomp& mc, const state& s, int dir);
    static wave wavespeed(const prim& W_L, const prim& W_R, const metriccomp& mc, const state& s, int dir);
private:
    // intermediate states
    struct starstt {
        cons U;
        prim W;
        double ptot;
    };
    static starstt xstate(const prim& W, const cons& U, const cons& F, const metriccomp& mc, double S, double Sx, int dir);
    static cons xxstate(const starstt& SL, const starstt& SR, const metriccomp& mc, double Sx, double SDx, int dir);
    // intermediate fluxes
    static cons physflux(const prim& W, const metriccomp& mc, int dir);
    static cons starflux(const cons& F, const cons& U, const cons& Ux, double S);
    static double fastws(const prim& W, const metriccomp& mc, const state stt, int dir);
    // magnetic field and pressure
    static std::array<double,4> Bfl(const prim& W, const metriccomp& mc);
    static double ptot(const prim& W, const metriccomp& mc) {
        return W.p+W.b2/2;
    }
};