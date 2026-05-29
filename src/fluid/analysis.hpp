#pragma once
#include <cmath>
#include <vector>
#include <array>
#include <functional>
#include "cell.hpp"
#include "grid.hpp"

// this document contains the structures and classes relating to the
// quantitative analysis of the gas.

using namespace grid;
namespace analysis {

// quantitative results bundle
struct batch {
    double t; // bundle time
    // conservation quantities
    double mass; // total mass
    double energy; // total energy
    double L_pol; // total angular momentum about pole
    // accretion quantities
    double Mdot; // mass accretion rate
    double Edot; // energy accretion rate
    double Ldot; // angular momentum accretion rate
    double Phi_B; // horizon magnetic flux
    double MAD; // magnetic arrest parameter
    // thermal quantities
    double magE; // magnetic energy
    double thermE; // thermal energy
    double L_BZ; // blandford-znajek luminosity
    double eta_BZ; // blandford-znajek efficiency
    // plasma structure quantities
    double Pbeta; // plasma beta parameter
    double alpha_ss; // shakura-sunyaev alpha
    double maxreyn; // maxwell-reynolds stress ratio
    double Hscal; // scale height
};
struct bundle {
    double t; // bundle time
    // analysis quantities
    double mass; // total mass
    double Mdot; // mass accretion rate
    double MAD; // magnetic arrest parameter
    double L_BZ; // blandford-znajek power
    double alpha_ss; // shakura-sunyaev alpha
};

class analyser {
public:
    double vol_int(const grid::amrtree& tree, const std::function<double(const cell&, double)> f);
    double surf_int(const grid::amrtree& tree, const double r_surf, std::function<double(const cell&, double)> f);
    batch batch_(const amrtree& tree, double t);
    bundle bundle_(const amrtree& tree, double t);
    
    double mass(const grid::amrtree& tree);
    double energy(const grid::amrtree& tree);
    double L_pol(const grid::amrtree& tree);

    double Mdot(const grid::amrtree& tree, double hrz);
    double Edot(const grid::amrtree& tree, double hrz);
    double Ldot(const grid::amrtree& tree, double hrz);
    double Phi_B(const grid::amrtree& tree, double hrz);
    double MAD(const grid::amrtree& tree, double hrz);

    double magE(const grid::amrtree& tree);
    double thermE(const grid::amrtree& tree);
    double L_BZ(const grid::amrtree& tree, double hrz);
    double eta_BZ(const grid::amrtree& tree, double hrz);

    double Pbeta(const grid::amrtree& tree);
    double alpha_ss(const grid::amrtree& tree, double rho_min);
    double maxreyn(const grid::amrtree& tree, double rho_min);
    double Hscal(const grid::amrtree& tree, double rs);
};

}
