#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>

#include "src/fluid/metric.hpp"
#include "src/fluid/state.hpp"
#include "src/fluid/prim.hpp"
#include "src/fluid/cons.hpp"
#include "src/fluid/cell.hpp"
#include "src/fluid/grid.hpp"
#include "src/fluid/hlld.hpp"
#include "src/fluid/recon.hpp"
#include "src/fluid/initial.hpp"
#include "src/fluid/ct.hpp"
#include "src/fluid/rk2.hpp"
#include "src/fluid/data.hpp"

// this is the main document for the fluid side of the project, and
// contains the iteration loop. this document is most heavily
// commented for user-friendliness. If there remain any points of
// confusion, consult our extensive user manual/developer's guide.

// this project was a long effort between Vincent Leung, Haris Forde,
// Archimedes Lentzos, and Dhruv Mathur, in no particular order.

// define types
using namespace grid;
using namespace torus;
using namespace integ;

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
    const double r_horizon = 1.0+std::sqrt(1.0-a*a); // horizon radius 1.0
    constexpr double dom_lo = -50.0*M; // lower corner of domain
    constexpr double dom_hi = 50.0*M; // upper corner of domain
    constexpr int nqlt = 4; // number of root patches
    // floors
    constexpr double rho_floor = 1e-5; // density floor
    constexpr double eps_floor = 1e-7; // energy floor
    constexpr double r_floor_ref = 1.0; // reference radius for scaling
    // integration
    constexpr double cfl = 0.4; // courant-friedrichs-lewy number
    constexpr double t_end = 10.0; // end time in geometrised units, NOT FRAMES!!
    constexpr double max_steps = 1000; // step count hard limit (this is now frames)
}
using namespace params;

// main function
int main() {
    std::cout << "Initialising grid...\n";
    // construct the amr tree
    std::array<double,3> dom_l = {params::dom_lo,params::dom_lo,params::dom_lo};
    std::array<double,3> dom_h = {params::dom_hi,params::dom_hi,params::dom_hi};
    amrtree tree(dom_l,dom_h,params::nqlt,params::M,params::a,params::Q,params::Gamma);
    std::cout << params::nqlt*params::nqlt*params::nqlt << " patches have been instantiated.\n";
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
                }
            }
        }
    }
    std::cout << "Maximum density across the entire grid:" << rho_max << "\n";
    // initialise magnetic field
    std::cout << "Initialising magnetic field...\n";
    for(const auto& p : tree.quilt) {
        init::B_pot_init(*p,tree.mtr,rho_max);
    }
    // convert primitives to conserveds in all cells
    std::cout << "Converting primitive variables to conserved variables...\n";
    for(const auto& p : tree.quilt) {
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    cell& c = p->cell_(i,j,k);
                    c.U = tree.cnsv.ptoc(c.W,c.r,c.th);
                }
            }
        }
    }
    // set up runge-kutta integrator and time loop
    integ::rk2integrator rk(params::cfl);
    double t = 0.0;
    int step = 0;
    // bool ok = true;
    // auto timer_start = std::chrono::steady_clock::now();
    // start integrator and god help your computer
    std::cout << "Starting integration...\n";
    while(t<params::t_end && step<params::max_steps) {
        double dt = rk.step(tree);
        t += dt;
        step++;
    }
    std::cout << "Integrator completed this many timesteps: ";
    return step;
}