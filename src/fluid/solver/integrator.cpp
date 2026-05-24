#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include "hlld.hpp"
#include "recon.hpp"
#include "ct.hpp"
#include "grid.hpp"
#include "rk2.hpp"

// this document contains the methods relating to integration.
using namespace integ;
using namespace grid;

// flux divergence
cons rk2integrator::div_flux(const grid::patch& p, int i, int j, int k) {
    double dx = p.dx(), dy = p.dy(), dz = p.dz();
    double sg = p.cell_(i,j,k).mtr.comp(p.cell_(i,j,k).r,p.cell_(i,j,k).th).sqrtdetg;
    // get face fluxes
    const cons& FxR = p.get_Fx(i,j,k);
    const cons& FxL = p.get_Fx(i-1,j,k);
    const cons& FyR = p.get_Fy(i,j,k);
    const cons& FyL = p.get_Fy(i,j-1,k);
    const cons& FzR = p.get_Fz(i,j,k);
    const cons& FzL = p.get_Fz(i,j,k-1);
    // compute divergences in each conserved variable
    cons dU;
    dU.D = -((FxR.D-FxL.D)/dx+(FyR.D-FyL.D)/dy+(FzR.D-FzL.D)/dz);
    dU.tau = -((FxR.tau-FxL.tau)/dx+(FyR.tau-FyL.tau)/dy+(FzR.tau-FzL.tau)/dz);
    for(int i=0; i<3; i++) {
        dU.S[i] = -((FxR.S[i]-FxL.S[i])/dx+(FyR.S[i]-FyL.S[i])/dy+(FzR.S[i]-FzL.S[i])/dz);
        dU.B[i] = 0.0; // div B = 0 ^^
    }

    return dU;
}

// geometric source terms
cons rk2integrator::source(const patch& p, int i, int j, int k, const metric& mtr, const state& stt) {
    // metric quantities
    const cell& c = p.cell_(i,j,k);
    double r = c.r, th = c.th;
    metriccomp mc = c.mtr.comp(r,th);
    // christoffel symbols
    double Gam[4][4][4];
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            for(int k=0; k<4; k++) {
                Gam[i][j][k] = mc.Gamma[i][j][k];
            }
        }
    }
    // construct some useful quantities
    double v2 = 0.0;
    double vlow[3] = {}, Blow[3] = {};
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            vlow[i] += mc.gam[i][j]*c.W.v[j];
            Blow[i] += mc.gam[i][j]*c.W.B[j];
        }
        v2 += vlow[i]*c.W.v[i];
    }
    v2 = std::min(v2,1-1e-10);
    double ltz2 = 1.0/(1.0-v2);
    double ltz = std::sqrt(ltz2);
    // magnetic things
    double Bv = 0.0, Bsq = 0.0;
    for(int i=0; i<3; i++) {
        Bv += Blow[i]*c.W.v[i];
        Bsq += Blow[i]*c.W.B[i];
    }
    double b2 = Bv*Bv+Bsq/ltz2;
    // thermodynamic state quantities
    double rho = c.W.rho;
    double eps = c.W.eps;
    double px = stt.press(rho,eps);
    double h = stt.enth(rho,eps);

    // 4-velocity (contravariant and covariant)
    double u4[4];
    u4[0] = ltz/mc.alpha;
    for(int i=0; i<3; i++) {
        u4[i+1] = ltz*(c.W.v[i]-mc.beta[i]/mc.alpha);
    }
    double u_4[4] = {0.0,0.0,0.0,0.0};
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            u_4[i] += mc.g[i][j]*u4[j];
        }
    }
    // magnetic 4-vector
    double b4[4];
    b4[0] = ltz*Bv/mc.alpha;
    for(int i=0; i<3; i++) {
        b4[i+1] = (c.W.B[i]+mc.alpha*b4[0]*u4[i+1])/ltz;
    }
    double b_4[4] = {0.0,0.0,0.0,0.0};
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            b_4[i] += mc.g[i][j]*b4[j];
        }
    }

    // stress-energy tensor
    double T44[4][4];
    double rhohb2 = rho*h+b2;
    for(int m=0; m<4; m++) {
        for(int n=0; n<4; n++) {
            T44[m][n] = rhohb2*u4[m]*u4[n]+px*mc.g_inv[m][n]-b4[m]*b4[n];
        }
    }
    // one index lowered
    double T4_4[4][4] = {};
    for(int m=0; m<4; m++) {
        for(int n=0; n<4; n++) {
            for(int s=0; s<4; s++) {
                T4_4[m][n] = mc.g[n][s]*T44[m][s];
            }
        }
    }
    // lowered christoffels
    double Gam_[4][4][4] = {};
    for(int n=0; n<4; n++) {
        for(int k=0; k<4; k++) {
            for(int l=0; l<4; l++) {
                for(int s=0; s<4; s++) {
                    Gam_[n][k][l] += mc.g[n][s]*Gam[s][k][l];
                }
            }
        }
    }
    // source terms
    double src[4] = {};
    for(int m=0; m<4; m++) {
        for(int n=0; n<4; n++) {
            for(int l=0; l<4; l++) {
                src[m] += T44[n][l]*Gam_[m][n][l];
            }
        }
        src[m] *= mc.sqrtdetg;
    }
    
    // assign values
    cons S;
    S.D = 0.0; // no mass source
    S.tau = src[0];
    S.S[0] = src[1]; S.S[1] = src[2]; S.S[2] = src[3];
    S.B[0] = S.B[1] = S.B[2] = 0.0; // handled elsewhere
    
    return S;
}

// compute the required timestep which obeys CFL
double rk2integrator::dtcomp(const amrtree& tree, double cfl) {
    state stt = tree.stt;
    double dt_min = std::numeric_limits<double>::max();
    for(const auto& p : tree.quilt) {
        // only iterate over leaves (highest-level refinement cells)
        if(!p->leaf) continue;
        double dx = std::min({p->dx(),p->dy(),p->dz()});
        double v_max = 0.0;
        // maximum signal speed in every cell
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    const cell& c = p->cell_(i,j,k);
                    double cs2 = stt.cs2(c.W.rho,c.W.eps);
                    double cs = std::sqrt(cs2);
                    double vmag = std::sqrt(c.W.v[0]*c.W.v[0]+c.W.v[1]*c.W.v[1]+c.W.v[2]*c.W.v[2]);
                    v_max = std::max(v_max,vmag+cs);
                }
            }
        }
        if(v_max>1e-14) {
            dt_min = std::min(dt_min,cfl*dx/v_max);
        }
    }
    return dt_min;
}
// runge-kutta stages
void rk2integrator::rkstg(amrtree& tree, double dt, int stage) {
    // fill ghosts
    for(auto& p : tree.quilt) {
        if(!p->leaf) continue;
        tree.ghosts(p.get());
    }
    // reconstruct primitives everywhere
    for(auto& p : tree.quilt) {
        for(int i=-ghost; i<block+ghost; i++) {
            for(int j=-ghost; j<block+ghost; j++) {
                for(int k=-ghost; k<block+ghost; k++) {
                    cell& c = p->cell_(i,j,k);
                    prim pv;
                    prim& PV = pv;
                    bool ok = tree.cnsv.ctop(c.U,c.r,c.th,PV);
                    if(ok) c.W = PV;
                    else {
                        prim fl = tree.pvfs(c.r,c.th);
                        fl.B[0] = c.W.B[0]; fl.B[1] = c.W.B[1]; fl.B[2] = c.W.B[2];
                        c.W = fl;
                        c.U = tree.cnsv.ptoc(c.W,c.r,c.th);
                    }
                }
            }
        }
    }
    // compute fluxes
    for(auto& p : tree.quilt) {
        p->fluxcomp(tree.mtr,tree.stt);
    }
    // emfs and magnetic field
    for(auto& p : tree.quilt) {
        constrans::emfcomp(*p);
        constrans::Bfupdate(*p,dt);
        constrans::f2cB(*p);
    }
    // conserved variable update
    for(auto& p : tree.quilt) {
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    cell& c = p->cell_(i,j,k);
                    cons dU = rk2integrator::div_flux(*p,i,j,k)+rk2integrator::source(*p,i,j,k,tree.mtr,tree.stt);
                    if(stage==0) {
                        c.dU = c.U; // update accumulator
                        c.U = c.U+dt*dU;
                    } else {
                        c.U = (c.dU+c.U+dt*dU)*0.5;
                    }
                    metriccomp mc = c.mtr.comp(c.r,c.th);
                    c.U.B[0] = mc.sqrtdetg*c.W.B[0];
                    c.U.B[1] = mc.sqrtdetg*c.W.B[1];
                    c.U.B[2] = mc.sqrtdetg*c.W.B[2];
                }
            }
        }
    }
}
// runge-kutta time step
double rk2integrator::step(amrtree& tree) {
    double dt = dtcomp(tree,cfl);
    // call runge-kutta stages
    rkstg(tree,dt,0);
    rkstg(tree,dt,1);
    // enforce floors
    for(auto& p : tree.quilt) {
        p->floors(tree,*p,tree.stt,tree.mtr);
    }
    return dt;
}