#include <cmath>
#include <vector>
#include <array>
#include "grid.hpp"
#include "initial.hpp"

// this document contains the methods for initialising variables
// subject to the appropriate constraints.

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
// time component of 4-velocity for given angular momentum
static double comp_utsq (const metriccomp& mc, double l) {
    // metric components
    double gtt = mc.g[0][0];
    double gtph = mc.g[0][3];
    double gphph = mc.g[3][3];

    double num = gtph*gtph-gtt*gphph;
    double den = gphph+2.0*l*gtph+l*l*gtt;
    // degeneracy check
    if(std::abs(den)<1e-14) return -1.0;

    return num/den;
}
// torus effective potential
static double torus_pot (const metriccomp& mc, double l, double W_in) {
    double utsq = comp_utsq(mc,l);
    if(utsq<=0.0) return -1e30; // degeneracy if outside region 
    return 0.5*std::log(std::abs(utsq))-W_in;
}

// torus initialisation
void init::fm_init(amrtree& tree) {
    // metric parameters
    double M = tree.mtr.M;
    double a = tree.mtr.a;
    double Gamma = tree.stt.gamma;
    // rotation parameters for isco
    double Z1 = 1+pow(1-(a/M)*(a/M),1.0/3.0)*(pow(1+a/M,1.0/3.0)+pow(1-a/M,1.0/3.0));
    double Z2 = std::sqrt(3*(a/M)*(a/M)-Z1*Z1);
    // inner and outer torus limits
    double r_in = 1.5*M*(3+Z2-std::sqrt((3-Z1)*(3+Z1+2*Z2)));
    double r_max = 2.25*r_in;
    // density scale so that \rho_{max}=1 in code units
    double K = 0.01*pow(M,Gamma-1.0);
    double rho_tgt = 1.0;

    // angular momentum at pressure maximum
    metriccomp mc = tree.mtr.comp(r_max,M_PI/2.0);
    // metric components
    double gtt = mc.g[0][0];
    double gtph = mc.g[0][3];
    double gphph = mc.g[3][3];
    // finite central difference for metric derivatives
    double dr = 0.005*M;
    metriccomp mc_p = tree.mtr.comp(r_max+dr,M_PI/2.0);
    metriccomp mc_m = tree.mtr.comp(r_max-dr,M_PI/2.0);
    double dgtt = (mc_p.g[0][0]-mc_m.g[0][0])/(2*dr);
    double dgtph = (mc_p.g[0][3]-mc_m.g[0][3])/(2*dr);
    double dgphph = (mc_p.g[3][3]-mc_m.g[3][3])/(2*dr);
    // angular velocity
    double Omega_K = (-dgtph+std::sqrt(dgtph*dgtph-dgtt*dgphph))/dgphph;
    // angular momentum
    double l0 = -(gtph+Omega_K*gphph)/(gtt+Omega_K*gtph);

    // inner radius potential
    metriccomp mc_in = tree.mtr.comp(r_in,M_PI/2.0);
    double utsq_in = comp_utsq(mc_in,l0);
    if(utsq_in<=0.0) {
        std::cerr << "Fishbone-Moncrief torus initialisation threw back an error at inner edge: u_t^2 is nonpositive! Check inner radius.\n";
        return;
    }
    double W_in = 0.5*std::log(std::abs(utsq_in));

    // find maximum density for normalisation
    double rho_max = 0.0;
    for(const auto& p : tree.quilt) {
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    cell& c = p->cell_(i,j,k);
                    double r = c.r; double th = c.th;
                    // skip cells too near axis or horizon
                    if(r<1.05*tree.mtr.M || std::abs(std::sin(th))<0.01) continue;
                    metriccomp mc = tree.mtr.comp(r,th);
                    double W = torus_pot(mc,l0,W_in);
                    if(W>0.0) {
                        double rho = pow((Gamma-1.0)*W/(Gamma*K),1.0/(Gamma-1.0));
                        rho_max = std::max(rho_max,rho);

                    }
                }
            }
        }
    }
    if(rho_max<1e-14) {
        std::cerr << "Fishbone-Moncrief torus initialisation threw back an error: No torus cells found! Check initialisation parameters.\n";
        return;
    }
    double rho_scal = rho_tgt/rho_max;

    // set primitives
    for(const auto& p : tree.quilt) {
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    cell& c = p->cell_(i,j,k);
                    c.W = tree.pvfs(c.r,c.th);
                    double r = c.r; double th = c.th;
                    // default values are floors
                    prim W_p = tree.pvfs(r,th);
                    // horizon protections
                    double r_H = tree.mtr.M*(1.0+std::sqrt(1.0-(tree.mtr.a/tree.mtr.M)*(tree.mtr.a/tree.mtr.M)));
                    if(r<1.05*r_H || std::abs(std::sin(th))<0.01) {
                        c.W = W_p; continue;
                    }
                    metriccomp mc = tree.mtr.comp(r,th);
                    double W_pot = torus_pot(mc,l0,W_in);
                    // values inside torus
                    if(W_pot>0.0) {
                        // density and energy
                        double rho = rho_scal*pow((Gamma-1.0)*W_pot/(Gamma*K),1.0/(Gamma-1.0));
                        double prs = K*pow(rho,Gamma);
                        double eps = prs/((Gamma-1.0)*rho);
                        W_p.rho = rho; W_p.eps = eps;
                        // velocity
                        double utsq = comp_utsq(mc,l0);
                        double ut = std::sqrt(std::abs(utsq));
                        double Omega = -(mc.g[0][3]+l0*mc.g[0][0])/(mc.g[3][3]+l0*mc.g[0][3]);
                        double uphys = Omega*ut;
                        // coordinate velocity with kerr-schild lapse and shift
                        W_p.v[0] = 0.0; W_p.v[1] = 0.0; W_p.v[2] = 0.0;
                        // convert to cartesian
                        double vphc = (uphys/ut-mc.beta[2]/mc.alpha); // d\phi/dt
                        double x = c.xc; double y = c.yc;
                        double rcyl = std::sqrt(x*x+y*y); // cylindrical coordinate radius
                        if(rcyl>1e-10) {
                            W_p.v[0] = -vphc*y;
                            W_p.v[1] = vphc*x;
                        }
                        W_p.v[2] = 0.0;
                    }
                    c.W = W_p;
                }
            }
        }
    }
    std::cout << "Fishbone-Moncrief torus initialisation: L_0 = " << l0 << "    r_in = " << r_in << "    r_max = " << r_max << "    rho_max = " << rho_tgt << "\n";
}

// magnetic field main initialisation function
void init::B_pot_init(patch& p, const metric& mtr, double glmx_rho) {
    double dx = p.dx(), dy = p.dy(), dz = p.dz();
    // determine maximum density over patch
    double rho_max = glmx_rho;
    if(rho_max<=1e-14) return; // if patch not a toroidal patch, return without doing anything else
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
                double Ax_kp = A_vpot(x_xkp,y_xkp,z_xkp,rho_xkp,rho_max)[2];
                double Ax_km = A_vpot(x_xkm,y_xkm,z_xkm,rho_xkm,rho_max)[2];
                // z edge
                auto [x_zip,y_zip,z_zip] = zedge_pos(p,i,j,k);
                auto [x_zim,y_zim,z_zim] = zedge_pos(p,i-1,j,k);
                double rho_zip = edge_rhoz(p,i,j,k);
                double rho_zim = edge_rhoz(p,i-1,j,k);
                double Az_ip = A_vpot(x_zip,y_zip,z_zip,rho_zip,rho_max)[1];
                double Az_im = A_vpot(x_zim,y_zim,z_zim,rho_zim,rho_max)[1];
                // update
                p.Bfy[p.Bfy_idx(i,j,k)] = (Ax_kp-Ax_km)/dz-(Az_ip-Az_im)/dx;

                // Bfz update
                // x edge
                auto [x_yip,y_yip,z_yip] = xedge_pos(p,i,j,k);
                auto [x_yim,y_yim,z_yim] = xedge_pos(p,i-1,j,k);
                double rho_yip = edge_rhox(p,i,j,k);
                double rho_yim = edge_rhox(p,i-1,j,k);
                double Ay_ip = A_vpot(x_yip,y_yip,z_yip,rho_yip,rho_max)[2];
                double Ay_im = A_vpot(x_yim,y_yim,z_yim,rho_yim,rho_max)[2];
                // y edge
                auto [x_xjp,y_xjp,z_xjp] = yedge_pos(p,i,j,k);
                auto [x_xjm,y_xjm,z_xjm] = yedge_pos(p,i,j-1,k);
                double rho_xjp = edge_rhoy(p,i,j,k);
                double rho_xjm = edge_rhoy(p,i,j-1,k);
                double Ax_jp = A_vpot(x_xjp,y_xjp,z_xjp,rho_xjp,rho_max)[1];
                double Ax_jm = A_vpot(x_xjm,y_xjm,z_xjm,rho_xjm,rho_max)[1];
                // update
                p.Bfz[p.Bfz_idx(i,j,k)] = (Ay_ip-Ay_im)/dx-(Ax_jp-Ax_jm)/dy;
            }
        }
    }
    // average face to cell magnetic fields for reconstruction later
    constrans::f2cB(p);
}