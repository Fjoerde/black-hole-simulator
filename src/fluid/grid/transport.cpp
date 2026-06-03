#include <cmath>
#include <vector>
#include <array>
#include "ct.hpp"

// this document contains the methods to ensure the magnetic field is
// transported with zero divergence.
using namespace grid;

// compute emfs at edges
void constrans::emfcomp(patch& p) {
    for(int i=-ghost; i<block+ghost-1; i++) {
        for(int j=-ghost; j<block+ghost-1; j++) {
            for(int k=-ghost; k<block+ghost-1; k++) {
                // get x-component of emf
                double Fy_Bz_k = p.get_Fy_B(2,i,j,k);
                double Fy_Bz_kp = p.get_Fy_B(2,i,j,k+1);
                double Fz_By_j = p.get_Fz_B(1,i,j,k);
                double Fz_By_jp = p.get_Fz_B(1,i,j+1,k);
                p.EMFx[p.EMFx_idx(i,j,k)] = (Fy_Bz_k+Fy_Bz_kp-Fz_By_j-Fz_By_jp)/4.0;
                // get y-component of emf
                double Fx_Bz_k = p.get_Fx_B(2,i,j,k);
                double Fx_Bz_kp = p.get_Fx_B(2,i,j,k+1);
                double Fz_Bx_i = p.get_Fz_B(0,i,j,k);
                double Fz_Bx_ip = p.get_Fz_B(0,i+1,j,k);
                p.EMFy[p.EMFy_idx(i,j,k)] = (Fz_Bx_i+Fz_Bx_ip-Fx_Bz_k-Fx_Bz_kp)/4.0;
                // get z-component of emf
                double Fx_By_j = p.get_Fx_B(1,i,j,k);
                double Fx_By_jp = p.get_Fx_B(1,i,j+1,k);
                double Fy_Bx_i = p.get_Fy_B(0,i,j,k);
                double Fy_Bx_ip = p.get_Fy_B(0,i+1,j,k);
                p.EMFz[p.EMFz_idx(i,j,k)] = (Fx_By_j+Fx_By_jp-Fy_Bx_i-Fy_Bx_ip)/4.0;
            }
        }
    }
}
// update magnetic field at faces
void constrans::Bfupdate(patch& p, double dt, const std::vector<double>& Bfx_n, const std::vector<double>& Bfy_n, const std::vector<double>& Bfz_n, double kappa) {
    double dx = p.dx(), dy = p.dy(), dz = p.dz();
    for(int i=-ghost+1; i<block+ghost-1; i++) {
        for(int j=-ghost+1; j<block+ghost-1; j++) {
            for(int k=-ghost+1; k<block+ghost-1; k++) {
                int idx_x = p.Bfx_idx(i,j,k);
                int idx_y = p.Bfy_idx(i,j,k);
                int idx_z = p.Bfz_idx(i,j,k);
                // p.Bfx[p.Bfx_idx(i,j,k)] = 0.0; p.Bfy[p.Bfy_idx(i,j,k)] = 0.0; p.Bfz[p.Bfz_idx(i,j,k)] = 0.0;
                // electric field derivatives
                double dEz_dy = (p.EMFz[p.EMFz_idx(i,j,k)]-p.EMFz[p.EMFz_idx(i,j-1,k)])/dy;
                double dEy_dz = (p.EMFy[p.EMFy_idx(i,j,k)]-p.EMFy[p.EMFy_idx(i,j,k-1)])/dz;
                double dEz_dx = (p.EMFz[p.EMFz_idx(i,j,k)]-p.EMFz[p.EMFz_idx(i-1,j,k)])/dx;
                double dEx_dz = (p.EMFx[p.EMFx_idx(i,j,k)]-p.EMFx[p.EMFx_idx(i,j,k-1)])/dz;
                double dEy_dx = (p.EMFy[p.EMFy_idx(i,j,k)]-p.EMFy[p.EMFy_idx(i-1,j,k)])/dx;
                double dEx_dy = (p.EMFx[p.EMFx_idx(i,j,k)]-p.EMFx[p.EMFx_idx(i,j-1,k)])/dy;
                // update magnetic fields at faces
                p.Bfx[p.Bfx_idx(i,j,k)] = (1.0-kappa)*Bfx_n[idx_x]+kappa*p.Bfx[idx_x]+kappa*dt*(dEz_dy-dEy_dz);
                p.Bfy[p.Bfy_idx(i,j,k)] = (1.0-kappa)*Bfy_n[idx_y]+kappa*p.Bfy[idx_y]+kappa*dt*(dEx_dz-dEz_dx);
                p.Bfz[p.Bfz_idx(i,j,k)] = (1.0-kappa)*Bfz_n[idx_z]+kappa*p.Bfz[idx_z]+kappa*dt*(dEy_dx-dEx_dy);
            }
        }
    }
}
// average face magnetic fields to cell centre magnetic fields
void constrans::f2cB(patch& p) {
    for(int i=-ghost+1; i<block+ghost; i++) {
        for(int j=-ghost+1; j<block+ghost; j++) {
            for(int k=-ghost+1; k<block+ghost; k++) {
                cell& c = p.cell_(i,j,k);
                c.W.B[0] = (p.Bfx[p.Bfx_idx(i,j,k)]+p.Bfx[p.Bfx_idx(i-1,j,k)])/2.0;
                c.W.B[1] = (p.Bfy[p.Bfy_idx(i,j,k)]+p.Bfy[p.Bfy_idx(i,j-1,k)])/2.0;
                c.W.B[2] = (p.Bfz[p.Bfz_idx(i,j,k)]+p.Bfz[p.Bfz_idx(i,j,k-1)])/2.0;
                // update conserved variables as well by scaling by
                // determinant of the metric
                double sg = c.mtr.comp(c.r,c.th).sqrtdetg;
                c.U.B[0] = sg*c.W.B[0]; c.U.B[1] = sg*c.W.B[1]; c.U.B[2] = sg*c.W.B[2];
            }
        }
    }
}
// diagnostic to check \div B is approximately 0
double constrans::maxdivB(const patch& p) {
    double maxdiv = 0;
    double dx = p.dx(), dy = p.dy(), dz = p.dz();
    for(int i=-ghost+1; i<block+ghost-1; i++) {
        for(int j=-ghost+1; j<block+ghost-1; j++) {
            for(int k=-ghost+1; k<block+ghost-1; k++) {
                double divB = (p.Bfx[p.Bfx_idx(i,j,k)]-p.Bfx[p.Bfx_idx(i-1,j,k)])/dx+(p.Bfy[p.Bfy_idx(i,j,k)]-p.Bfy[p.Bfy_idx(i,j-1,k)])/dy+(p.Bfz[p.Bfz_idx(i,j,k)]-p.Bfz[p.Bfz_idx(i,j,k-1)])/dz;
                maxdiv = std::max(maxdiv,std::abs(divB));
            }
        }
    }
    return maxdiv;
}