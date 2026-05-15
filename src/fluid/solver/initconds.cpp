#include <cmath>
#include <vector>
#include <array>
#include "../initial.hpp"

// this document contains the methods for initialising variables
// subject to the appropriate constraints.

// magnetic field initialised from curl of vector potential
std::array<double,3> init::A_vpot(double x, double y, double z, double rho, double rho_max) {
    // torus initialisation, so only A_z component nonzero
    double Az = std::max(rho/rho_max-0.2,0.0);
    return {0.0,0.0,Az};
}
// edge densities
static double edge_rhox(const patch& p, int i, int j, int k) {
    return (p.cell_(i,j,k).W.rho+p.cell_(i,j+1,k).W.rho+p.cell_(i,j,k+1).W.rho+p.cell_(i,j+1,k+1).W.rho)/4;
}
static double edge_rhoy(const patch& p, int i, int j, int k) {
    return (p.cell_(i,j,k).W.rho+p.cell_(i+1,j,k).W.rho+p.cell_(i,j,k+1).W.rho+p.cell_(i+1,j,k+1).W.rho)/4;
}
static double edge_rhoz(const patch& p, int i, int j, int k) {
    return (p.cell_(i,j,k).W.rho+p.cell_(i+1,j,k).W.rho+p.cell_(i,j+1,k).W.rho+p.cell_(i+1,j+1,k).W.rho)/4;
}
// physical coordinates of edges
static std::array<double,3> xedge_pos(const patch& p, int i, int j, int k) {
    return {p.xedge[i]+p.dx()/2,p.yedge[j+1],p.zedge[k+1]};
}
static std::array<double,3> yedge_pos(const patch& p, int i, int j, int k) {
    return {p.xedge[i+1]/2,p.yedge[j]+p.dy(),p.zedge[k+1]};
}
static std::array<double,3> zedge_pos(const patch& p, int i, int j, int k) {
    return {p.xedge[i+1]/2,p.yedge[j+1],p.zedge[k]+p.dz()};
}

// magnetic field main initialisation function
void init::B_pot_init(patch& p, const metric& mtr, double glmx_rho) {
    double dx = p.dx(), dy = p.dy(), dz = p.dz();
    // determine maximum density over patch
    double rho_max = 0.0;
    for(int i=0; i<block; i++) {
        for(int j=0; j<block; j++) {
            for(int k=0; k<block; k++) {
                rho_max = std::max(rho_max,p.cell_(i,j,k).W.rho);
            }
        }
    }
    // magnetic updates
    for(int i=-ghost; i<block+ghost; i++) {
        for(int j=-ghost; j<block+ghost; j++) {
            for(int k=-ghost; k<block+ghost; k++) {
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