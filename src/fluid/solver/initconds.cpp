#include <cmath>
#include <vector>
#include <array>
#include "grid.hpp"
#include "initial.hpp"

// this document contains the methods for initialising variables
// subject to the appropriate constraints.

// NOTE: THIS CODE IS EXTREMELY SENSITIVE AND A HUGE PAIN TO DEBUG!
// this file is a graveyard of a week's worth of abject suffering...

using namespace grid;
using namespace torus;

// magnetic field initialised from curl of vector potential
std::array<double,3> init::A_vpot(double x, double y, double z, double rho, double rho_max) {
    // torus initialisation, so only A_phi component nonzero
    double Ap = std::max(rho/rho_max-0.2,0.0);
    double phi = atan2(y,x);
    double theta = std::acos(z/std::sqrt(x*x+y*y+z*z));
    return {-Ap*std::sin(phi)*std::sin(theta),Ap*std::cos(phi)*std::sin(theta),0.0};
}
// edge densities, averaged from 4 surrounding cells
static double edge_rhox(const patch& p, int i, int j, int k) {
    int j1 = std::min(j+1,block+ghost-1);
    int k1 = std::min(k+1,block+ghost-1);
    return (p.cell_(i,j,k).W.rho+p.cell_(i,j1,k).W.rho+p.cell_(i,j,k1).W.rho+p.cell_(i,j1,k1).W.rho)/4;
}
static double edge_rhoy(const patch& p, int i, int j, int k) {
    int i1 = std::min(i+1,block+ghost-1);
    int k1 = std::min(k+1,block+ghost-1);
    return (p.cell_(i,j,k).W.rho+p.cell_(i1,j,k).W.rho+p.cell_(i,j,k1).W.rho+p.cell_(i1,j,k1).W.rho)/4;
}
static double edge_rhoz(const patch& p, int i, int j, int k) {
    int i1 = std::min(i+1,block+ghost-1);
    int j1 = std::min(j+1,block+ghost-1);
    return (p.cell_(i,j,k).W.rho+p.cell_(i1,j,k).W.rho+p.cell_(i,j1,k).W.rho+p.cell_(i1,j1,k).W.rho)/4;
}
// physical coordinates of edges
static std::array<double,3> xedge_pos(const patch& p, int i, int j, int k) {
    const cell& c = p.cell_(i,j,k);
    return {c.xc,c.yc+p.dy()/2.0,c.zc+p.dz()/2.0};
}
static std::array<double,3> yedge_pos(const patch& p, int i, int j, int k) {
    const cell& c = p.cell_(i,j,k);
    return {c.xc+p.dx()/2.0,c.yc,c.zc+p.dz()/2.0};
}
static std::array<double,3> zedge_pos(const patch& p, int i, int j, int k) {
    const cell& c = p.cell_(i,j,k);
    return {c.xc+p.dx()/2.0,c.yc+p.dy()/2.0,c.zc};
}
// // time component of 4-velocity for given angular momentum
// static double comp_utsq (const amrtree& tree, const metriccomp& mc, double r, double l) {
//     // metric components and useful quantities
//     double gtt = mc.g[0][0];
//     double gtr = mc.g[0][1];
//     double gtph = mc.g[0][3];
//     double grr = mc.g[1][1];
//     double grph = mc.g[1][3];
//     double gphph = mc.g[3][3];
//     double M = tree.mtr.M;
//     double a = tree.mtr.a;
//     double Q = tree.mtr.Q;
//     // intermediate quantities
//     double Delta = r*r+a*a+Q*Q-2.0*M*r;
//     double lambda = (2.0*M*r-Q*Q)/Delta;
//     double mu = a/Delta;
//     double gtt_eff = gtt+2.0*lambda*gtr+lambda*lambda*grr;
//     double gtph_eff = gtph+lambda*mu*grr+mu*gtr+lambda*grph;
//     double gphph_eff = gphph+2.0*mu*grph+mu*mu*grr;
//     double num = gtph_eff*gtph_eff-gtt_eff*gphph_eff;
//     double den = gphph_eff+2.0*l*gtph_eff+l*l*gtt_eff;
//     // degeneracy check
//     if(std::abs(den)<1e-14) return -1.0;
//     std::cout << "Initialisation diagnostic for u_{t}^2: denominator = " << den << "\n";
//     return num/den;
// }

// static double comp_utsq(const amrtree& tree, double l, double r) {
//     double M = tree.mtr.M;
//     double a = tree.mtr.a;
//     double Q = tree.mtr.Q;
//     // helper lambdas for boyer-lindquist
//     auto gtt_BL_f = [&](double r) {
//         return -(1.0-(2.0*M*r-Q*Q)/(r*r));
//     };
//     auto gtph_BL_f = [&](double r) {
//         return -(2.0*M*r-Q*Q)*a/(r*r);
//     };
//     auto gphph_BL_f = [&](double r) {
//         return r*r+a*a+(2*M*r-Q*Q)*a*a/(r*r);
//     };
//     double num = gtph_BL_f(r)*gtph_BL_f(r)-gtt_BL_f(r)*gphph_BL_f(r);
//     double den = gphph_BL_f(r)+2.0*l*gtph_BL_f(r)+l*l*gtt_BL_f(r);
//     return num/den;
// }

// static double comp_utsq(const metriccomp& mc, double l) {
//     // circular orbit in kerr-schild has only t and \phi components
//     double gtt = mc.g[0][0];
//     double gtp = mc.g[0][3];
//     double gpp = mc.g[3][3];
//     // intermediate quantities
//     double num = gtp*gtp-gtt*gpp;
//     double den = gpp+2.0*l*gtp+l*l*gtt;
//     if(std::abs(den)<1e-10) return -1.0;
//     return num/den;
// }
// old keplerian L_0
// static double comp_L0K(double r, double M, double a, double Q) {
//     const double num = r*r+a*a+Q*Q*(a/std::sqrt(M*r)-1.0)-2.0*a*std::sqrt(M*r);
//     const double den = std::sqrt(r*r*r)+a*std::sqrt(M)+Q*Q*(1.0/std::sqrt(r)-a*std::sqrt(M)/(r*std::sqrt(M*r)))-2.0*M*std::sqrt(r);
//     if(std::abs(den)<1e-10) {
//         throw std::runtime_error("Fishbone-Moncrief torus angular momentum calculations threw back an error: The denominator vanished, so r_{max} is at or inside the ISCO! Check initialisation parameters.");
//     }
//     return num/den;
// }

// angular mometum
static double L0_comp(const metric& mtr, double r_max, double M) {
    const double dr = 1e-4*M;
    // metric and perturbations around r_max for central differences
    const metriccomp mc0 = mtr.comp(r_max,M_PI/2.0);
    const metriccomp mcp = mtr.comp(r_max+dr,M_PI/2.0);
    const metriccomp mcm = mtr.comp(r_max-dr,M_PI/2.0);
    double gtt = mc0.g[0][0];
    double gtp = mc0.g[0][3];
    double gpp = mc0.g[3][3];
    // metric derivatives
    const double dgtt = (mcp.g[0][0]-mcm.g[0][0])/(2.0*dr);
    const double dgtp = (mcp.g[0][3]-mcm.g[0][3])/(2.0*dr);
    const double dgpp = (mcp.g[3][3]-mcm.g[3][3])/(2.0*dr);
    // intermediate quantities
    double A = -gtp*(gpp*(dgtt*dgpp+2*dgtp*dgtp)+gtt*dgpp*dgpp)+std::sqrt((dgtp*dgtp-dgtt*dgpp)*pow((gpp*(gpp*dgtt-gtt*dgpp)+2*gtp*gtp*dgpp-2*gtp*gpp*dgtp),2))+gpp*gpp*dgtt*dgtp+gtt*gpp*dgtp*dgpp+2*gtp*gtp*dgtp*dgpp;
    double B = dgtt*(gpp*gpp*dgtt+4*gtp*gtp*dgpp-4*gtp*gpp*dgtp)-2*gtt*(gpp*(dgtt*dgpp-2*dgtp*dgtp)+2*gtp*dgtp*dgpp)+gtt*gtt*dgpp*dgpp;
    if(std::abs(B)<1e-14) {
        throw std::runtime_error("LNRF angular momentum initialisation threw back an error: denominator is zero, so r_max is inside the ergosphere!");
    }
    return A/B;
}

// torus effective potential
// static double torus_pot(const metriccomp& mc, double l, double W_in) {
//     double utsq = comp_utsq(mc,l);
//     if(utsq<=1e-14) return -1e30; // degeneracy if outside region 
//     return 0.5*std::log(utsq);
// }

// locally non-rotating frame (LNRF) decomposition
struct lnrf {
    double e2nu;
    double e2psi;
    double enu;
    double epsi;
    double omega;
};
static lnrf lnrf_(const metriccomp& mc) {
    const double gtt = mc.g[0][0];
    const double gtp = mc.g[0][3];
    const double gpp = mc.g[3][3];

    lnrf lp;
    lp.e2nu = gtp*gtp/gpp-gtt;
    lp.e2psi = gpp;
    lp.enu = std::sqrt(lp.e2nu);
    lp.epsi = std::sqrt(lp.e2psi);
    lp.omega = -gtp/gpp;

    return lp;
}

// torus potentials
static double lnh_pot(const metriccomp& mc, double l, double W_in) {
    const lnrf lp = lnrf_(mc);
    if(lp.e2nu<=0.0) return -1e30; // ergosphere, so unstable
    return -0.5*std::sqrt((4*l*l*lp.e2nu)/lp.e2psi+1)+0.5*std::log((std::sqrt((4*l*l*lp.e2nu)/lp.e2psi+1)+1)/lp.e2nu)-l*lp.omega+W_in;
}
static double W_in_(const metriccomp& mc, double l) {
    const lnrf lp = lnrf_(mc);
    if(lp.e2nu<=0.0) return -1e30; // ergosphere, so unstable
    return 0.5*std::sqrt((4*l*l*lp.e2nu)/lp.e2psi+1)-0.5*std::log((std::sqrt((4*l*l*lp.e2nu)/lp.e2psi+1)+1)/lp.e2nu)+l*lp.omega;
}

// thermodynamic helpers
struct therm {double rho, p, eps;};
static therm disctherm(double lnH, double Gamma, double K) {
    therm th;
    if(lnH<=0.0) {
        th.rho = th.p = th.eps = 0.0;
        return th;
    }
    const double H = std::exp(lnH);
    const double arg = (Gamma-1)*(H-1)/(Gamma*K);
    if(arg<=0.0) {
        th.rho = th.p = th.eps = 0.0;
        return th;
    }
    th.rho = std::pow(arg,1.0/(Gamma-1.0));
    th.p = K*std::pow(arg,Gamma/(Gamma-1.0));
    th.eps = th.p/((Gamma-1.0)*th.rho);
    return th;
}

// 3-velocity normalised properly by 4-velocity
static std::array<double,3> velcomp(const metriccomp& mc, const lnrf& lp, double l, double xc, double yc) {
    const double u_phi = std::sqrt(0.5*(std::sqrt(1.0+4.0*l*l*lp.e2nu/lp.e2psi)-1.0)); // covariant
    const double uphi = u_phi/lp.epsi+(lp.omega/lp.enu)*std::sqrt(1.0+u_phi*u_phi); // contravariant
    // contravariant time component by normalisation
    const double gtt = mc.g[0][0];
    const double gtp = mc.g[0][3];
    const double gpp = mc.g[3][3];
    const double B = 2.0*gtp*uphi;
    const double C = gpp*uphi*uphi+1.0;
    const double D = B*B-4.0*gtt*C;
    if(D<0.0) return {0.0,0.0,0.0};
    // future u^t > 0 as gtt < 0 outside ergosphere
    const double ut = (-B-std::sqrt(D))/(2.0*gtt);
    const double Omega = uphi/ut;
    const double vphi = Omega-mc.beta[2]/mc.alpha;
    const double rcyl = std::sqrt(xc*xc+yc*yc);
    if(rcyl<1e-10) return {0.0,0.0,0.0};

    return {-vphi*yc,vphi*xc,0.0};
}

// torus initialisation
void init::fm_init(amrtree& tree) {
    // metric parameters
    double M = tree.mtr.M;
    double a = tree.mtr.a;
    double Q = tree.mtr.Q;
    double Gamma = tree.stt.gamma;
    // useful radii
    const double r_max = 12.0*M; // <-- <-- FREE PARAMETER
    const double r_in = 4.5*M; // <-- <-- FREE PARAMETER
    double r_H = M+std::sqrt(M*M-a*a-Q*Q);
    // double l0 = comp_L0K(r_max,M,a,Q);
    // double l0 = 12.0; // <-- <-- FREE PARAMETER
    // rotation parameters for isco
    // double Z1 = 1.0+pow(1.0-(a/M)*(a/M),1.0/3.0)*(pow(1+a/M,1.0/3.0)+pow(1-a/M,1.0/3.0));
    // double Z2 = std::sqrt(3*(a/M)*(a/M)+Z1*Z1);
    
    // std::cout << "Equator values of W\n";
    // for (double rs = 1.1*r_H; rs <= 25.0*M; rs += 0.5*M) {
    //     metriccomp mcs = tree.mtr.comp(rs, M_PI/2.0);
    //     double us = comp_utsq(mcs, l0);
    //     if (us <= 0.0 || !std::isfinite(us)) {
    //         std::cout << "r = " << rs << ", INVALID\n";
    //         continue;
    //     }
    //     double Ws = 0.5*std::log(us);
    //     std::cout << "r = "  << rs << "W = " << Ws << "\n";
    // }
    
    // solve for radius of maximum pressure by bisection
    // double r_max_tgt = 14.0*M;
    // double W_max = W_in, r_max = r_max_tgt;
    // for(double rs = 1.05*r_in; rs < 25.0*M; rs += 0.01*M) {
    //     metriccomp mcs = tree.mtr.comp(rs, M_PI/2.0);
    //     double us = comp_utsq(mcs, l0);
    //     if(us<=1e-10 || !std::isfinite(us)) continue;
    //     double Ws = 0.5*std::log(us);
    //     if(Ws>W_max) {W_max = Ws; r_max = rs;}
    // }
    
    // solve for inner torus radius by bisection
    // double W_saddle = 1e30, r_in = 1.05*r_H;
    // for(double rs = 1.05*r_H; rs<r_max-0.1*M; rs += 0.05*M) {
    //     metriccomp mcs = tree.mtr.comp(rs, M_PI/2.0);
    //     double us = comp_utsq(mcs, l0);
    //     if(us <= 1e-10 || !std::isfinite(us)) continue;
    //     double Ws = 0.5*std::log(us);
    //     if(Ws<W_saddle) {W_saddle=Ws; r_in=rs;}
    // }
    // metriccomp mcmax = tree.mtr.comp(r_max,M_PI/2.0);
    // const double utsq_max = comp_utsq(mcmax,l0);
    // if(utsq_max<=0.0 || !std::isfinite(utsq_max)) {
    //     std::cerr << "init::fm_init threw up an error: u_{t}^2 is nonpositive at r_max! Check r_max or L_0.\n";
    //     return;
    // }
    // // inner radius potential
    // metriccomp mc_in = tree.mtr.comp(r_in,M_PI/2.0);
    // double utsq_in = comp_utsq(mc_in,l0);
    // if(utsq_in<=0.0) {
    //     std::cerr << "Fishbone-Moncrief torus initialisation threw back an error at inner edge: u_t^2 is nonpositive! Check maximum radius.\n";
    //     return;
    // }
    // double W_in = 0.5*std::log(comp_utsq(mc_in,l0));
    // std::cout << "=== Finding minimum l0 for valid torus ===\n";
    // for (double l_test = 3.0; l_test <= 10.0; l_test += 0.1) {
    //     // find where W first becomes valid (utsq > 0) scanning outward
    //     double r_valid_start = -1.0;
    //     double W_max_test = -1e30;
    //     for (double rs = 1.1*r_H; rs < 30.0*M; rs += 0.05*M) {
    //         metriccomp mcs = tree.mtr.comp(rs, M_PI/2.0);
    //         double us = comp_utsq(mcs, l_test);
    //         if (us <= 0.0) continue;
    //         if (r_valid_start < 0.0) r_valid_start = rs;
    //         double Ws = 0.5*std::log(us);
    //         W_max_test = std::max(W_max_test, Ws);
    //     }
    //     // check if W has a proper maximum (not just at the valid boundary)
    //     // by checking if W at r_valid_start+1M < W_max_test
    //     if (r_valid_start > 0.0) {
    //         metriccomp mc_check = tree.mtr.comp(r_valid_start + 1.0*M, M_PI/2.0);
    //         double u_check = comp_utsq(mc_check, l_test);
    //         if (u_check > 0.0) {
    //             double W_check = 0.5*std::log(u_check);
    //             if (W_max_test > W_check + 0.01) {
    //                 std::cout << "l0=" << l_test
    //                         << " has valid maximum, r_valid_start="
    //                         << r_valid_start << "\n";
    //             }
    //         }
    //     }
    // }
    // double W_max = 0.5*std::log(utsq_max);
    // double WFmin = W_max;
    // double rFmin = r_max;
    // for(double rsc=1.05*r_H; rsc<r_max; rsc+=0.05*M) {
    //     const metriccomp mcs = tree.mtr.comp(rsc,M_PI/2.0);
    //     const double utsq_s = comp_utsq(mcs,l0);
    //     if(utsq_s<=0.0 || !std::isfinite(utsq_s)) continue;
    //     const double Ws = 0.5*std::log(utsq_s);
    //     if(Ws<WFmin) {
    //         WFmin = Ws;
    //         rFmin = rsc;
    //     }
    // }
    // double rl = rFmin;
    // double rh = r_max-1e-6*r_H;
    // for(int i=0; i<1000; i++) {
    //     const double rm = 0.5*(rl+rh);
    //     const metriccomp mcm = tree.mtr.comp(rm,M_PI/2.0);
    //     double utsq_mid = comp_utsq(mcm,l0);
    //     if(utsq_mid<=0.0) {rl=rm; continue;}
    //     const double W_mid = 0.5*std::log(utsq_mid);
    //     (W_mid<W_max)? rl = rm : rh = rm;
    //     if(rh-rl<1e-10*M) break;
    // }
    // const double r_in = 0.5*(rl+rh);
    // const double W_in = W_max;
    // double r_in = 2.5*M*(3.0+Z2-std::sqrt((3.0-Z1)*(3.0+Z1+2.0*Z2)));
    // density scale so that \rho_{max}=1 in code units
    double K = 0.001*pow(M,Gamma-1.0);
    double rho_tgt = 1.0;
    // std::cout << "ISCO diagnostic: \nr_in = " << r_in << "\nr_max = " << r_max << "\n";

    // angular momentum at pressure maximum
    // metriccomp mc = tree.mtr.comp(r_max,M_PI/2.0);
    // metric components
    // double gtt = mc.g[0][0];
    // double gtph = mc.g[0][3];
    // double gphph = mc.g[3][3];
    // finite central difference for metric derivatives
    // double dr = 0.01*M;
    // metriccomp mc_p = tree.mtr.comp(r_max+dr,M_PI/2.0);
    // metriccomp mc_m = tree.mtr.comp(r_max-dr,M_PI/2.0);
    // double dgtt = (mc_p.g[0][0]-mc_m.g[0][0])/(2.0*dr);
    // double dgtph = (mc_p.g[0][3]-mc_m.g[0][3])/(2.0*dr);
    // double dgphph = (mc_p.g[3][3]-mc_m.g[3][3])/(2.0*dr);
    // angular velocity
    // double Omega_K = (-dgtph+std::sqrt(dgtph*dgtph-dgtt*dgphph))/dgphph;
    // double Omega_K = (std::sqrt(M))/(std::sqrt(r_max*r_max*r_max)+a*std::sqrt(M));
    // angular momentum
    // double l0 = -(gtph+Omega_K*gphph)/(gtt+Omega_K*gtph);
    // double l0 = (r_max*r_max-2*a*std::sqrt(M*r_max)+a*a)/(std::sqrt(r_max*r_max*r_max)-2*r_max*std::sqrt(M)+a*std::sqrt(M));
    // double l0 = -gtph/gtt;
    // double num = gtph*gtph-gtt*gphph;
    // double den = 2*gtph*dgtph-dgtt*gphph-gtt*dgphph;
    // double A = den*gtt-num*dgtt;
    // double B = den*gtph-num*dgtph;
    // double C = den*gphph-num*dgphph;
    // double D = B*B-A*C;
    // if(D<0) {
    //     std::cerr << "Fishbone-Moncrief torus initialisation threw up an error: there is no real solution for angular momentum! Check r_max.";
    //     return;
    // }
    // double l0_p = (-B+std::sqrt(D))/A;
    // double l0_m = (-B-std::sqrt(D))/A;
    // double l0 = (l0_p>0)? l0_p : l0_m;

    // angular momentum, computed with boyer-lindquist components
    // explicity because the theoretical astrophysics papers are
    // frustratingly inconsistent and cannot agree on a metric, and
    // practically all use boyer-lindqust despite the vast majority
    // of grmhd simulations being written in kerr-schild because it's
    // the OBVIOUS CHOICE but they don't seem to care -_- ah well
    // double Delta = r_max*r_max-2*M*r_max+a*a+Q*Q;
    // double r2 = r_max*r_max;
    // double gtt_BL = -(1.0-(2.0*M*r_max-Q*Q)/r2);
    // double gtph_BL = -(2.0*M*r_max-Q*Q)*a/r2;
    // double gphph_BL = r2+a*a+(2.0*M*r_max-Q*Q)*a*a/r2;
    // double dH_BL = 2.0*M/r2-2.0*(2.0*M*r_max-Q*Q)/(r2*r_max);
    // double dgtt = -dH_BL;
    // double dr2 = 0.001*M;
    // // helper lambdas
    // auto gtt_BL_f = [&](double r) {
    //     return -(1.0-(2.0*M*r-Q*Q)/(r*r));
    // };
    // auto gtph_BL_f = [&](double r) {
    //     return -(2.0*M*r-Q*Q)*a/(r*r);
    // };
    // auto gphph_BL_f = [&](double r) {
    //     return r*r+a*a+(2*M*r-Q*Q)*a*a/(r*r);
    // };
    // // numerical derivatives
    // double dgtt_BL = (gtt_BL_f(r_max+dr2)-gtt_BL_f(r_max-dr2))/(2.0*dr2);
    // double dgtph_BL = (gtph_BL_f(r_max+dr2)-gtph_BL_f(r_max-dr2))/(2.0*dr2);
    // double dgphph_BL = (gphph_BL_f(r_max+dr2)-gphph_BL_f(r_max-dr2))/(2.0*dr2);
    // // equation solving
    // double BLnum = gtph_BL*gtph_BL-gtt_BL*gphph_BL;
    // double BLden = 2*gtph_BL*dgtph_BL-dgtt_BL*gphph_BL-dgphph_BL*gtt_BL;
    // double A = BLden*gtt_BL-BLnum*dgtt_BL;
    // double B = BLden*gtph_BL-BLnum*dgtph_BL;
    // double C = BLden*gphph_BL-BLnum*dgphph_BL;
    // double D = B*B-A*C;
    // if(D<0.0) {
    //     std::cerr << "Fishbone-Moncrief torus initialisation threw back an error: Boyer-Lindquist determinant is negative, so there exist no real solutions for angular momentum! Check maximum radius.\n";
    // }
    // double l0_pBL = (-B+std::sqrt(D))/A;
    // double l0_mBL = (-B-std::sqrt(D))/A;
    // // angular velocity
    // auto Omega_K = [&](double l) {
    //     return -(gtph_BL+l*gtt_BL)/(gphph_BL+l*gtph_BL);
    // };
    // double l0 = (l0_pBL>0.0 && Omega_K(l0_pBL)>0.0)? l0_pBL : l0_mBL;
    // std::cout << "Initialisation angular momentum diagnostic: L_0 = " << l0 << "\n";
    // angular momentum
    const double l0 = L0_comp(tree.mtr,r_max,M);
    std::cout << "Initialisation diagnostic: L_0 = " << l0 << "\n";
    // inner potential
    const metriccomp mc_in = tree.mtr.comp(r_in,M_PI/2.0);
    const double W_in = W_in_(mc_in,l0);
    std::cout << "Initialisation diagnostic: W_in = " << W_in << "\n";
    // diagnostic: ensure ln(h) > 0 at r_max
    const metriccomp mc_max = tree.mtr.comp(r_max,M_PI/2.0);
    const double lnh_max = lnh_pot(mc_max,l0,W_in);
    std::cout << "Initialisation diagnostic: at r = r_max, ln(h) = " << lnh_max << "\n";
    if(lnh_max<=0.0) {
        std::cerr << "Fishbone-Moncrief torus initialisation threw back an error in init::fm_init: ln(h) <= 0.0 at r = r_max! Check initialisation parameters r_in and r_max, as the torus cannot initialise.\n";
        return;
    }
    // std::cout << "Initialisation diagnostic at inner radius: g_{tt} = " << mc_in.g[0][0] << "\n";
    // std::cout << "Initialisation diagnostic at inner radius: g_{tphi} = " << mc_in.g[0][3] << "\n";
    // std::cout << "Initialisation diagnostic at inner radius: g_{phiphi} = " << mc_in.g[3][3] << "\n";
    // std::cout << "Initialisation diagnostic at inner radius: u_{t_{in}}^2 = " << utsq_in << "\n";
    // std::cout << "Initialisation diagnostic at inner radius: Omega_K = " << Omega_K << "\n";
    // std::cout << "Initialisation diagnostic at inner radius: W_{in} = " << W_in << "\n";
    // std::cout << "Initialisation diagnostic at maximum radius: W_{max} = " << torus_pot(mcmax,l0,W_in) << "\n";
    // find maximum density for normalisation
    double rho_max = 0.0;
    for(const auto& p : tree.quilt) {
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    cell& c = p->cell_(i,j,k);
                    double r = c.r; double th = c.th;
                    // skip cells too near axis or horizon, or with bad coordinate evaluations
                    if(!std::isfinite(r) || !std::isfinite(th) || r<=1.05*r_H || std::abs(std::sin(th))<0.01) continue;
                    metriccomp mc = tree.mtr.comp(r,th);
                    // double W = torus_pot(mc,l0,W_in);
                    // double DelW = W-W_in;
                    // double h = std::exp(DelW);
                    // std::cout << "init::fm_init diagnostic for h: " << h << "\n";
                    // std::cout << "init::fm_init diagnostic for torus potential: " << W_in << "\n";
                    // if(DelW>0.0 && std::isfinite(W)) {
                    //     double rho = pow((Gamma-1.0)*(h-1)/(Gamma*K),1.0/(Gamma-1.0));
                    //     if(std::isfinite(rho)) rho_max = std::max(rho_max,rho);
                    //     // std::cout << "init::fm_init diagnostic for density: " << rho << "\n";
                    // }
                    const double lnH = lnh_pot(mc,l0,W_in);
                    if(lnH<=0.0 || !std::isfinite(lnH)) continue;
                    const therm thm = disctherm(lnH,Gamma,K);
                    if(std::isfinite(thm.rho)) {
                        rho_max = std::max(rho_max,thm.rho);
                    }
                }
            }
        }
    }
    if(rho_max<1e-14) {
        std::cerr << "Fishbone-Moncrief torus initialisation threw back an error: No torus cells found! Check initialisation parameters.\n";
        return;
    }
    // // diagnostic to check whether torus was generated correctly
    // int torc = 0;
    // int atm_cells = 0;
    // double rho_max_torus = 0.0;
    // double lnh_max_found = 0.0;

    // // recompute l0 and W_in for the diagnostic
    // const double l0_diag  = L0_comp(tree.mtr, r_max, M);
    // const metriccomp mc_in_diag = tree.mtr.comp(r_in, M_PI/2.0);
    // const double W_in_diag = W_in_(mc_in_diag, l0_diag);
    // for (const auto& p : tree.quilt) {
    //     for (int i = 0; i < block; ++i)
    //     for (int j = 0; j < block; ++j)
    //     for (int k = 0; k < block; ++k) {
    //         const cell& c = p->cell_(i,j,k);
    //         const double r = c.r, th = c.th;
    //         // skip unphysical cells
    //         if (!std::isfinite(r) || !std::isfinite(th)) continue;
    //         if (r < 1.05*r_H || std::abs(std::sin(th)) < 0.01) continue;
    //         const metriccomp mc = tree.mtr.comp(r, th);
    //         const lnrf lp = lnrf_(mc);
    //         if (lp.e2nu <= 0.0) continue;
    //         const double lnh = lnh_pot(mc, l0_diag, W_in_diag);
    //         if (lnh > 0.0 && std::isfinite(lnh)) {
    //             torc++;
    //             rho_max_torus = std::max(rho_max_torus, c.W.rho);
    //             lnh_max_found = std::max(lnh_max_found, lnh);
    //         } else {
    //             atm_cells++;
    //         }
    //     }
    // }

    // std::cout << "Torus cells (ln(h)>0): " << torc << "\n";
    // std::cout << "Atmosphere cells: " << atm_cells << "\n";
    // std::cout << "Max density in torus: " << rho_max_torus << "\n";
    // std::cout << "Max ln(h) found: " << lnh_max_found << "\n";
    // std::cout << "Expected ln(h)_max: " << lnh_max << "\n";
    // std::cout << "init::fm_init diagnostic for density: " << rho_max << "\n";
    double rho_scal = rho_tgt/rho_max;
    // set primitives
    for(const auto& p : tree.quilt) {
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    cell& c = p->cell_(i,j,k);
                    // c.W = tree.pvfs(c.r,c.th);
                    double r = c.r; double th = c.th;
                    // default values are floors
                    prim W_p = tree.pvfs(r,th);
                    // horizon protections
                    if(!std::isfinite(r) || !std::isfinite(th) || r<=1.05*r_H || std::abs(std::sin(th))<0.01) {
                        c.W = W_p; continue;
                    }
                    metriccomp mc = tree.mtr.comp(r,th);
                    const double lnH = lnh_pot(mc,l0,W_in);
                    if(lnH<=0.0 || !std::isfinite(lnH)) {
                        c.W = W_p; continue;
                    }
                    // double DelW = W_pot-W_in;
                    // values inside torus
                    // if(DelW>0.0) {
                    //     // density and energy
                    //     double h = std::exp(DelW);
                    //     double rho = rho_scal*pow((Gamma-1.0)*(h-1)/(Gamma*K),1.0/(Gamma-1.0));
                    //     double prs = K*pow(rho,Gamma);
                    //     double eps = prs/((Gamma-1.0)*rho);
                    //     W_p.rho = rho; W_p.eps = eps;
                    //     // velocity
                    //     double utsq = comp_utsq(mc,l0);
                    //     double ut = std::sqrt(std::abs(utsq));
                    //     double Omega = -(mc.g[0][3]+l0*mc.g[0][0])/(mc.g[3][3]+l0*mc.g[0][3]);
                    //     double uphys = Omega*ut;
                    //     // coordinate velocity with kerr-schild lapse and shift
                    //     W_p.v[0] = 0.0; W_p.v[1] = 0.0; W_p.v[2] = 0.0;
                    //     // convert to cartesian
                    //     double vphc = (uphys/ut-mc.beta[2]/mc.alpha); // d\phi/dt
                    //     double x = c.xc; double y = c.yc;
                    //     double rcyl = std::sqrt(x*x+y*y); // cylindrical coordinate radius
                    //     if(rcyl>1e-10) {
                    //         W_p.v[0] = -vphc*y;
                    //         W_p.v[1] = vphc*x;
                    //     }
                    //     W_p.v[2] = 0.0;
                    // }
                    const therm thm = disctherm(lnH,Gamma,K);
                    W_p.rho = rho_scal*thm.rho;
                    W_p.p = K*std::pow(W_p.rho,Gamma);
                    W_p.eps = W_p.p/((Gamma-1.0)*W_p.rho);

                    const lnrf lp = lnrf_(mc);
                    const auto vel = velcomp(mc,lp,l0,c.xc,c.yc);
                    W_p.v[0] = vel[0];
                    W_p.v[1] = vel[1];
                    W_p.v[2] = vel[2];

                    c.W = W_p;
                    // std::cout << "init::fm_init diagnostic : " << c.W.rho << "    " << c.W.eps << "    " << c.W.p << "\n";
                }
            }
        }
    }
    std::cout << "Fishbone-Moncrief torus initialisation:\nL_0 = " << l0 << "\nr_in = " << r_in << "\nr_max = " << r_max << "\nrho_max = " << rho_max << "\nW_in = " << W_in << "\nln(h)_max = " << lnh_max << "\n";
    return;
}

// magnetic field main initialisation function
void init::B_pot_init(patch& p, const metric& mtr, double glmx_rho) {
    double dx = p.dx(), dy = p.dy(), dz = p.dz();
    // determine maximum density over patch
    double rho_max = glmx_rho;
    if(rho_max<=1e-10) return; // if patch not a toroidal patch, return without doing anything
    for(int i=0; i<block; i++) {
        for(int j=0; j<block; j++) {
            for(int k=0; k<block; k++) {
                rho_max = std::max(rho_max,p.cell_(i,j,k).W.rho);
            }
        }
    }
    // magnetic updates
    for(int i=-ghost+1; i<block+ghost; i++) {
        for(int j=-ghost+1; j<block+ghost; j++) {
            for(int k=-ghost+1; k<block+ghost; k++) {
                // Bfx update
                // y edge
                auto [x_ykp,y_ykp,z_ykp] = yedge_pos(p,i,j,k);
                auto [x_ykm,y_ykm,z_ykm] = yedge_pos(p,i,j,k-1);
                double rho_ykp = edge_rhoy(p,i,j,k);
                double rho_ykm = edge_rhoy(p,i,j,k-1);
                double Ay_kp = A_vpot(x_ykp,y_ykp,z_ykp,rho_ykp,rho_max)[1];
                double Ay_km = A_vpot(x_ykm,y_ykm,z_ykm,rho_ykm,rho_max)[1];
                // z edge
                auto [x_zjp,y_zjp,z_zjp] = zedge_pos(p,i,j,k);
                auto [x_zjm,y_zjm,z_zjm] = zedge_pos(p,i,j-1,k);
                double rho_zjp = edge_rhoz(p,i,j,k);
                double rho_zjm = edge_rhoz(p,i,j-1,k);
                double Az_jp = A_vpot(x_zjp,y_zjp,z_zjp,rho_zjp,rho_max)[2];
                double Az_jm = A_vpot(x_zjm,y_zjm,z_zjm,rho_zjm,rho_max)[2];
                // update
                p.Bfx[p.Bfx_idx(i,j,k)] = (Az_jp-Az_jm)/dy-(Ay_kp-Ay_km)/dz;

                // Bfy update
                // x edge
                auto [x_xkp,y_xkp,z_xkp] = xedge_pos(p,i,j,k);
                auto [x_xkm,y_xkm,z_xkm] = xedge_pos(p,i,j,k-1);
                double rho_xkp = edge_rhox(p,i,j,k);
                double rho_xkm = edge_rhox(p,i,j,k-1);
                double Ax_kp = A_vpot(x_xkp,y_xkp,z_xkp,rho_xkp,rho_max)[0];
                double Ax_km = A_vpot(x_xkm,y_xkm,z_xkm,rho_xkm,rho_max)[0];
                // z edge
                auto [x_zip,y_zip,z_zip] = zedge_pos(p,i,j,k);
                auto [x_zim,y_zim,z_zim] = zedge_pos(p,i-1,j,k);
                double rho_zip = edge_rhoz(p,i,j,k);
                double rho_zim = edge_rhoz(p,i-1,j,k);
                double Az_ip = A_vpot(x_zip,y_zip,z_zip,rho_zip,rho_max)[2];
                double Az_im = A_vpot(x_zim,y_zim,z_zim,rho_zim,rho_max)[2];
                // update
                p.Bfy[p.Bfy_idx(i,j,k)] = (Ax_kp-Ax_km)/dz-(Az_ip-Az_im)/dx;

                // Bfz update
                // x edge
                auto [x_yip,y_yip,z_yip] = xedge_pos(p,i,j,k);
                auto [x_yim,y_yim,z_yim] = xedge_pos(p,i-1,j,k);
                double rho_yip = edge_rhox(p,i,j,k);
                double rho_yim = edge_rhox(p,i-1,j,k);
                double Ay_ip = A_vpot(x_yip,y_yip,z_yip,rho_yip,rho_max)[1];
                double Ay_im = A_vpot(x_yim,y_yim,z_yim,rho_yim,rho_max)[1];
                // y edge
                auto [x_xjp,y_xjp,z_xjp] = yedge_pos(p,i,j,k);
                auto [x_xjm,y_xjm,z_xjm] = yedge_pos(p,i,j-1,k);
                double rho_xjp = edge_rhoy(p,i,j,k);
                double rho_xjm = edge_rhoy(p,i,j-1,k);
                double Ax_jp = A_vpot(x_xjp,y_xjp,z_xjp,rho_xjp,rho_max)[0];
                double Ax_jm = A_vpot(x_xjm,y_xjm,z_xjm,rho_xjm,rho_max)[0];
                // update
                p.Bfz[p.Bfz_idx(i,j,k)] = (Ay_ip-Ay_im)/dx-(Ax_jp-Ax_jm)/dy;
            }
        }
    }
    // average face to cell magnetic fields for reconstruction later
    constrans::f2cB(p);
}