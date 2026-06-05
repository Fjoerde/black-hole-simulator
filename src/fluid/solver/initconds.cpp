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
    double Ap = std::max(rho/rho_max-0.02,0.0);
    double phi = atan2(y,x);
    double theta = std::acos(z/std::sqrt(x*x+y*y+z*z));
    return {-Ap*std::sin(phi)*std::sin(theta),Ap*std::cos(phi)*std::sin(theta),0.0};
}
// edge densities, averaged from 4 surrounding cells
static double edge_rhox(const patch& p, int i, int j, int k) {
    int j1 = std::min(j+1,block+ghost-1);
    int k1 = std::min(k+1,block+ghost-1);
    return (p.cell_(i,j,k).W.rho+p.cell_(i,j1,k).W.rho+p.cell_(i,j,k1).W.rho+p.cell_(i,j1,k1).W.rho)/4.0;
}
static double edge_rhoy(const patch& p, int i, int j, int k) {
    int i1 = std::min(i+1,block+ghost-1);
    int k1 = std::min(k+1,block+ghost-1);
    return (p.cell_(i,j,k).W.rho+p.cell_(i1,j,k).W.rho+p.cell_(i,j,k1).W.rho+p.cell_(i1,j,k1).W.rho)/4.0;
}
static double edge_rhoz(const patch& p, int i, int j, int k) {
    int i1 = std::min(i+1,block+ghost-1);
    int j1 = std::min(j+1,block+ghost-1);
    return (p.cell_(i,j,k).W.rho+p.cell_(i1,j,k).W.rho+p.cell_(i,j1,k).W.rho+p.cell_(i1,j1,k).W.rho)/4.0;
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
    th.p = K*std::pow(th.rho,Gamma);
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
double init::fm_init(amrtree& tree) {
    // metric parameters
    double M = tree.mtr.M;
    double a = tree.mtr.a;
    double Q = tree.mtr.Q;
    double Gamma = tree.stt.gamma;
    // useful quantities
    const double r_max = 12.0*M; // <-- <-- FREE PARAMETER
    const double r_in = 4.5*M; // <-- <-- FREE PARAMETER
    double r_H = M+std::sqrt(M*M-a*a-Q*Q);
    double K = 0.001*pow(M,Gamma-1.0);
    double rho_tgt = 1e-5;
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
    double lnh_min = 0.01*lnh_max;
    std::cout << "Initialisation diagnostic: at r = r_max, ln(h) = " << lnh_max << "\n";
    if(lnh_max<=0.0) {
        std::cerr << "Fishbone-Moncrief torus initialisation threw back an error in init::fm_init: ln(h) <= 0.0 at r = r_max! Check initialisation parameters r_in and r_max, as the torus cannot initialise.\n";
        return 0.0;
    }
    // find maximum density
    double rho_max = 0.0;
    for(const auto& p : tree.quilt) {
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    cell& c = p->cell_(i,j,k);
                    double r = c.r; double th = c.th;
                    // skip cells too near axis or horizon, or with bad coordinate evaluations
                    if(!std::isfinite(r) || !std::isfinite(th) || r<=1.5*r_H || std::abs(std::sin(th))<0.01) continue;
                    metriccomp mc = tree.mtr.comp(r,th);
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
        return 0.0;
    }
    double rho_max_raw = rho_max;
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
                    if(!std::isfinite(r) || !std::isfinite(th) || r<=1.5*r_H || std::abs(std::sin(th))<0.01) {
                        c.W = W_p; continue;
                    }
                    metriccomp mc = tree.mtr.comp(r,th);
                    const double lnH = lnh_pot(mc,l0,W_in);
                    if(lnH<=0.0 || !std::isfinite(lnH) || lnH<=lnh_min) {
                        c.W = W_p; continue;
                    }
                    const therm thm = disctherm(lnH,Gamma,K);
                    W_p.rho = thm.rho/rho_max; W_p.p = thm.p/pow(rho_max,Gamma); W_p.eps = W_p.p/((Gamma-1.0)*W_p.rho);

                    const lnrf lp = lnrf_(mc);
                    const auto vel = velcomp(mc,lp,l0,c.xc,c.yc);
                    W_p.v[0] = vel[0]; W_p.v[1] = vel[1]; W_p.v[2] = vel[2];
                    c.W = W_p;
                    // std::cout << "init::fm_init diagnostic : " << c.W.rho << "    " << c.W.eps << "    " << c.W.p << "\n";
                }
            }
        }
    }
    std::cout << "Fishbone-Moncrief torus initialisation:\nL_0 = " << l0 << "\nr_in = " << r_in << "\nr_max = " << r_max << "\nrho_max = " << rho_max << "\nW_in = " << W_in << "\nln(h)_max = " << lnh_max << "\n";
    return 1.0;
}

// magnetic field main initialisation function
void init::B_pot_init(patch& p, amrtree& tree, const metric& mtr, double glmx_rho) {
    double dx = p.dx(), dy = p.dy(), dz = p.dz();
    // determine maximum density over patch
    double rho_max_loc = 0.0;
    for(int i=0; i<block; i++) {
        for(int j=0; j<block; j++) {
            for(int k=0; k<block; k++) {
                rho_max_loc = std::max(rho_max_loc,p.cell_(i,j,k).W.rho);
            }
        }
    }
    if(rho_max_loc<0.01*glmx_rho) {
        for(auto& b : p.Bfx) b = 0.0;
        for(auto& b : p.Bfy) b = 0.0;
        for(auto& b : p.Bfz) b = 0.0;
        constrans::f2cB(p);
        return;
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
                double Ay_kp = A_vpot(x_ykp,y_ykp,z_ykp,rho_ykp,glmx_rho)[1];
                double Ay_km = A_vpot(x_ykm,y_ykm,z_ykm,rho_ykm,glmx_rho)[1];
                // z edge
                auto [x_zjp,y_zjp,z_zjp] = zedge_pos(p,i,j,k);
                auto [x_zjm,y_zjm,z_zjm] = zedge_pos(p,i,j-1,k);
                double rho_zjp = edge_rhoz(p,i,j,k);
                double rho_zjm = edge_rhoz(p,i,j-1,k);
                double Az_jp = A_vpot(x_zjp,y_zjp,z_zjp,rho_zjp,glmx_rho)[2];
                double Az_jm = A_vpot(x_zjm,y_zjm,z_zjm,rho_zjm,glmx_rho)[2];
                // update
                p.Bfx[p.Bfx_idx(i,j,k)] = (Az_jp-Az_jm)/dy-(Ay_kp-Ay_km)/dz;

                // Bfy update
                // x edge
                auto [x_xkp,y_xkp,z_xkp] = xedge_pos(p,i,j,k);
                auto [x_xkm,y_xkm,z_xkm] = xedge_pos(p,i,j,k-1);
                double rho_xkp = edge_rhox(p,i,j,k);
                double rho_xkm = edge_rhox(p,i,j,k-1);
                double Ax_kp = A_vpot(x_xkp,y_xkp,z_xkp,rho_xkp,glmx_rho)[0];
                double Ax_km = A_vpot(x_xkm,y_xkm,z_xkm,rho_xkm,glmx_rho)[0];
                // z edge
                auto [x_zip,y_zip,z_zip] = zedge_pos(p,i,j,k);
                auto [x_zim,y_zim,z_zim] = zedge_pos(p,i-1,j,k);
                double rho_zip = edge_rhoz(p,i,j,k);
                double rho_zim = edge_rhoz(p,i-1,j,k);
                double Az_ip = A_vpot(x_zip,y_zip,z_zip,rho_zip,glmx_rho)[2];
                double Az_im = A_vpot(x_zim,y_zim,z_zim,rho_zim,glmx_rho)[2];
                // update
                p.Bfy[p.Bfy_idx(i,j,k)] = (Ax_kp-Ax_km)/dz-(Az_ip-Az_im)/dx;

                // Bfz update
                // x edge
                auto [x_yip,y_yip,z_yip] = xedge_pos(p,i,j,k);
                auto [x_yim,y_yim,z_yim] = xedge_pos(p,i-1,j,k);
                double rho_yip = edge_rhox(p,i,j,k);
                double rho_yim = edge_rhox(p,i-1,j,k);
                double Ay_ip = A_vpot(x_yip,y_yip,z_yip,rho_yip,glmx_rho)[1];
                double Ay_im = A_vpot(x_yim,y_yim,z_yim,rho_yim,glmx_rho)[1];
                // y edge
                auto [x_xjp,y_xjp,z_xjp] = yedge_pos(p,i,j,k);
                auto [x_xjm,y_xjm,z_xjm] = yedge_pos(p,i,j-1,k);
                double rho_xjp = edge_rhoy(p,i,j,k);
                double rho_xjm = edge_rhoy(p,i,j-1,k);
                double Ax_jp = A_vpot(x_xjp,y_xjp,z_xjp,rho_xjp,glmx_rho)[0];
                double Ax_jm = A_vpot(x_xjm,y_xjm,z_xjm,rho_xjm,glmx_rho)[0];
                // update
                p.Bfz[p.Bfz_idx(i,j,k)] = (Ay_ip-Ay_im)/dx-(Ax_jp-Ax_jm)/dy;
            }
        }
    }
    // average face to cell magnetic fields for reconstruction later
    constrans::f2cB(p);
}