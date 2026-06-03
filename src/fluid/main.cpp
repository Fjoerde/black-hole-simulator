#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>

#include "rk2.hpp"
#include "grid.hpp"
#include "metric.hpp"
#include "state.hpp"
#include "prim.hpp"
#include "cons.hpp"
#include "cell.hpp"
#include "recon.hpp"
#include "hlld.hpp"
#include "initial.hpp"
#include "ct.hpp"
#include "data.hpp"
#include "analysis.hpp"

// this is the main document for the fluid side of the project, and
// contains the iteration loop. this document is most heavily
// commented for user-friendliness. If there remain any points of
// confusion, consult our extensive user manual/developer's guide.

// this project was a long effort between Haris Forde, Archimedes Lentzos, Vincent Leung, and
// Dhruv Mathur, in no particular order.

// define types
using namespace grid;
using namespace torus;
using namespace integ;
using namespace analysis;

// user-friendly operation parameters ^^
// DO NOT SET ALL THREE EQUAL TO true SIMULTANEOUSLY!

// when set to true, grid data is written to a binary file. usually
// set to false when running tests to improve run speed. DO NOT SET
// TO TRUE WHEN RUNNING ON THE HPC EXCEPT FOR DATA COLLECTION RUNS!
bool data_write = false;

// when set to true, the analyser runs and gets quantitative results.
// usually set to true for testing purposes. RECOMMENDED: Set this to
// false when executing data collection runs on the HPC in order to
// not clog up the logs (as results are printed after each timestep),
// and to speed up the code.
bool enable_analyser = true;

// when set to true, the full detailed analysis of the data is done
// and results are printed after each integration step. usually set
// to false except for analysis runs. RECOMMENDED: Likely to slow
// down code significantly, so do not set to true while data_write is
// set to true unless absolutely necessary.
bool analyse_full = true;

// when set to true, analysis results are packaged in a compact way
// so as not to clog up the HPC logs. RECOMMENDED: set this to true
// when running long tests on the HPC for your convenience. Note this
// only affects output format when analyse_full is set to true.
bool hpc_run = false;

// simulation parameters
// note: units are a complete mess, ah well it's fine, geometrised
// units make a lot of things easier elsewhere
// we use G = c = 1
namespace params {
    // black hole parameters
    constexpr double M = 1.0; // mass
    constexpr double a = 0.9; // spin
    constexpr double Q = 0.1; // charge
    // thermodynamic state
    constexpr double Gamma = 5.0/3.0; // adiabatic index
    // domain
    const double r_horizon = M+std::sqrt(M*M-a*a-Q*Q); // horizon radius
    constexpr double dom_lo = -25.0*M; // lower corner of domain
    constexpr double dom_hi = 25.0*M; // upper corner of domain
    constexpr int nqlt = 8; // number of root patches per dimension
    // floors
    constexpr double rho_floor = 1e-5; // density floor
    constexpr double eps_floor = 1e-7; // energy floor
    constexpr double r_floor_ref = 1.0; // reference radius for scaling
    // integration
    constexpr double cfl = 0.4; // courant-friedrichs-lewy number
    constexpr double t_end = 200.0; // end time in geometrised units, NOT FRAMES!!
    constexpr double max_steps = 50; // step count hard limit (this is now frames)
}
// import namespaces
using namespace params;
using torus::init;
using grid::patch;
using grid::amrtree;

// main function
int main() {
    std::cout << "Initialising grid...\n";
    // construct the amr tree
    std::array<double,3> dom_l = {dom_lo,dom_lo,dom_lo};
    std::array<double,3> dom_h = {dom_hi,dom_hi,dom_hi};
    grid::amrtree tree(dom_l,dom_h,nqlt,M,a,Q,Gamma);
    std::cout << nqlt*nqlt*nqlt << " patches have been instantiated.\n";
    // initalise torus
    std::cout << "Initialising torus...\n";
    init::fm_init(tree);
    
    // global rho_max for initialisation of magnetic field
    double rho_max = 0.0;
    for(const auto& p : tree.quilt) {
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    rho_max = std::max(rho_max,p->cell_(i,j,k).W.rho);
                    // std::cout << "main diagnostic : " << p->cell_(i,j,k).W.rho << "    " << p->cell_(i,j,k).W.eps << "    " << p->cell_(i,j,k).W.p << "\n";
                }
            }
        }
    }
    if(rho_max<1e-14) {
        std::cerr << "Density initialisation threw back an error: rho_max is zero after Fishbone-Moncrief torus initialisation attempt!";
        return 1;
    }
    // // diagnostic to check torus was created properly
    // int torc = 0;
    // double rhcheck = 0.0;
    // for(const auto& p : tree.quilt) {
    //     for(int i=0; i<block; i++) {
    //         for(int j=0; j<block; j++) {
    //             for(int k=0; k<block; k++) {
    //                 const cell& c = p->cell_(i,j,k);
    //                 const prim fl = tree.pvfs(c.r,c.th);
    //                 if(c.W.rho>1000*fl.rho) {
    //                     torc++;
    //                     rhcheck = std::max(rhcheck,c.W.rho);
    //                 }
    //             }
    //         }
    //     }
    // }
    // std::cout << "Torus cells in grid: " << ": " << torc << "\nMaximum density: " << rhcheck << "\n";
    // if(torc==0) {
    //    std::cerr << "Fishbone-Moncrief torus initialisation threw back an error: no torus cells initialised!\n";
    //    return 1;
    // }
    
    std::cout << "Maximum density across the entire grid: " << rho_max << "\n";
    // initialise magnetic field
    std::cout << "Initialising magnetic field...\n";
    double p_max = 0.0; double B2_max = 0.0;
    for(const auto& p : tree.quilt) {
        init::B_pot_init(*p,tree.mtr,rho_max);
        // std::cout << "Initial density diagnostic : " << p->cell_(3,3,3).W.rho << "\n";
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    const cell& c = p->cell_(i,j,k);
                    p_max = std::max(p_max,c.W.p);
                    double B2 = c.W.B[0]*c.W.B[0]+c.W.B[1]*c.W.B[1]+c.W.B[2]*c.W.B[2];
                    B2_max = std::max(B2_max,B2);
                }
            }
        }
    }
    // normalise magnetic field
    const double beta_tgt = 100.0; // standard plasma beta normalisation to 100;
    const double B_scal = std::sqrt(2.0*p_max/(beta_tgt*B2_max));
    for(const auto& p : tree.quilt) {
        for(int i=-ghost; i<block+ghost; i++) {
            for(int j=-ghost; j<block+ghost; j++) {
                for(int k=-ghost; k<block+ghost; k++) {
                    cell& c = p->cell_(i,j,k);
                    c.W.B[0] *= B_scal; c.W.B[1] *= B_scal; c.W.B[2] *= B_scal;
                    metriccomp mc = c.mtr.comp(c.r,c.th);
                    c.U.B[0] = mc.sqrtdetg*c.W.B[0]; c.U.B[1] = mc.sqrtdetg*c.W.B[1]; c.U.B[2] = mc.sqrtdetg*c.W.B[2];
                }
            }
        }
        for(auto& bx : p->Bfx) bx *= B_scal;
        for(auto& by : p->Bfy) by *= B_scal;
        for(auto& bz : p->Bfz) bz *= B_scal;
    }
    // convert primitives to conserveds in all cells
    std::cout << "Converting primitive variables to conserved variables...\n";
    for(const auto& p : tree.quilt) {
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    cell& c = p->cell_(i,j,k);
                    c.U = tree.cnsv.ptoc(c.W,c.r,c.th);
                    // std::cout << "hi" << c.U.D << "\n"; // credit to vincent for the absolutely goated "hi" method of debugging
                    // std::cout << "Reconstructed density diagnostic: " << c.W.rho << "\n";
                }
            }
        }
    }
    // set up runge-kutta integrator and time loop
    rk2integrator rk(params::cfl);
    double t = 0.0;
    int step = 0;
    // bool ok = true;
    // auto timer_start = std::chrono::steady_clock::now();
    // start integrator and god help your computer
    std::cout << "Starting integration...\n";
    tree.regrid();
    if(hpc_run) {
        std::cout << "--- ANALYSIS HEADERS:    t.   M.   E.   L_pol.   Mdot.   Edot.   Phi_B.   MAD.   magE.   thermE.   L_BZ.   eta_BZ.   Pbeta.   alpha_ss.   maxreyn.   Hscal.\n";
    }
    while(t<params::t_end && step<params::max_steps) {
        double dt = rk.step(tree);
        t += dt;
        step++;
        std::cout << "Integrating at time: t = " << t << ", with timestep: dt = " << dt << "\n";
        // for(auto& p : tree.quilt) {
        //     std::cout << "Evolution density diagnostic: " << p->cell_(4,4,4).W.rho << "    " << p->cell_(4,4,4).U.D << "\n";
        // }
        analysis::analyser anl;
        if(enable_analyser) {
            if(analyse_full) {
                auto b = anl.batch_(tree,t);
                if(hpc_run) {
                    std::cout << "--- Timestep: t = " << t << "    " << b.mass << "    " << b.energy << "    " << b.L_pol << "    " << b.Mdot << "    " << b.Edot << "    " << b.Phi_B << "    " << b.MAD << "    " << b.magE << "    " << b.thermE << "    " << b.L_BZ << "    " << b.eta_BZ << "    " << b.Pbeta << "    " << b.alpha_ss << "    " << b.maxreyn << "    " << b.Hscal << "\n";
                } else {
                    std::cout << "--- ANALYSIS BATCH for timestep: t = " << t << "\nMass: " << b.mass << "\nEnergy: " << b.energy << "\nPolar angular momentum: " << b.L_pol << "\nMass accretion rate: " << b.Mdot << "\nEnergy accretion rate: " << b.Edot << "\nMagnetic flux through horizon: " << b.Phi_B << "\nMagnetically arrested disc (MAD) parameter: " << b.MAD << "\nTotal magnetic energy: " << b.magE << "\nTotal thermal energy: " << b.thermE << "\nBlandford-Znajek power: " << b.L_BZ << "\nBlandford-Znajek eta (efficiency): " << b.eta_BZ << "\nPlasma beta parameter: " << b.Pbeta << "\nShakura-Sunyaev alpha parameter: " << b.alpha_ss << "\nMaxwell-Reynolds stress ratio: " << b.maxreyn << "\nScale height: " << b.Hscal << "\n";
                }
            } else {
                auto b = anl.bundle_(tree,t);
                std::cout << "~~~ ANALYSIS BUNDLE for timestep: t = " << t << "\nMass: " << b.mass << "\nMass accretion rate: " << b.Mdot << "\nMagnetically arrested disc (MAD) parameter: " << b.MAD << "\nBlandford-Znajek luminosity: " << b.L_BZ << "\nShakura-Sunyaev alpha parameter: " << b.alpha_ss << "\n";
            }
        }
        // magnetic field diagnostic
        double max_divB = 0;
        for(const auto& p : tree.quilt) {
            if(!p->leaf) continue;
            max_divB = std::max(max_divB,constrans::maxdivB(*p));
        }
        std::cout << "Maximum magnetic field divergence: " << max_divB << "\n";
    }
    std::cout << "Integrator finished with coordinate time " << t << " in " << step << " steps." << "\n";
    return 0;
}
