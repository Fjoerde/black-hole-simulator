#include <cmath>
#include <vector>
#include <array>
#include <functional>
#include "../grid.hpp"
#include "../cell.hpp"
#include "../analysis.hpp"

// this document contains the methods relating to quantitative
// analysis of the black hole and accretion disc.

using namespace analysis;
using namespace grid;
// helpers
// horizon radius
static double r_horizon(const grid::amrtree& tree) {
    double M = tree.mtr.M; double a = tree.mtr.a; double Q = tree.mtr.Q;
    return M+std::sqrt(M*M-a*a-Q*Q);
}
// valencia radial and azimuthal velocity
static double val_vr(const cell& c) {
    return (c.xc*c.W.v[0]+c.yc*c.W.v[1]+c.zc*c.W.v[2])/c.r;
}
static double val_vphi(const cell& c) {
    double R = std::sqrt(c.xc*c.xc+c.yc*c.yc);
    if(R<1e-10) return 0.0;
    return (c.xc*c.W.v[1]-c.yc*c.W.v[0])/(R*R);
}
// valencia radial and azimuthal magnetic field components
static double val_Br(const cell& c) {
    return (c.xc*c.W.B[0]+c.yc*c.W.B[1]+c.zc*c.W.B[2])/c.r;
}
static double val_Bphi(const cell& c) {
    double R = std::sqrt(c.xc*c.xc+c.yc*c.yc);
    if(R<1e-10) return 0.0;
    return (c.xc*c.W.B[1]-c.yc*c.W.B[0])/(R*R);
}
// total pressure
static double ptot(const cell& c) {
    return c.W.p+0.5*c.W.b2;
}
// integrate over a volume
double analyser::vol_int(const grid::amrtree& tree, std::function<double(const cell&, double)> f) {
    double integral = 0.0;
    for(const auto& pp : tree.quilt) {
        const grid::patch* p = pp.get();
        if(!p->leaf) continue;
        for(int i=0; i<grid::block; i++) {
            for(int j=0; j<grid::block; j++) {
                for(int k=0; k<grid::block; k++) {
                    const cell& c = p->cell_(i,j,k);
                    double sqrtdetg = c.vol/c.pvol;
                    integral += f(c,sqrtdetg)*c.vol;
                }
            }
        }
    }
    return integral;
}
// integrate over a surface
double analyser::surf_int(const grid::amrtree& tree, const double r_surf, std::function<double(const cell&, double)> f) {
    double integral = 0.0;
    for(const auto& pp : tree.quilt) {
        const grid::patch* p = pp.get();
        if(!p->leaf) continue;
        for(int i=0; i<grid::block; i++) {
            for(int j=0; j<grid::block; j++) {
                for(int k=0; k<grid::block; k++) {
                    const cell& c = p->cell_(i,j,k);
                    double shell = 0.5*std::sqrt(c.dx*c.dx+c.dy*c.dy+c.dz*c.dz);
                    if(std::abs(c.r-r_surf)>shell) continue;
                    double sqrtdetg = c.vol/c.pvol;
                    integral += f(c,sqrtdetg)*c.vol;
                }
            }
        }
    }
    return integral;
}

// conservation quantities
double analyser::mass(const grid::amrtree& tree) {
    return vol_int(tree,[](const cell& c, double sg) {
        return c.U.D;
    });
}
double analyser::energy(const grid::amrtree& tree) {
    return vol_int(tree,[](const cell& c, double sg) {
        return c.U.tau+c.U.D;
    });
}
double analyser::L_pol(const grid::amrtree& tree) {
    return vol_int(tree,[](const cell& c, double sg) {
        return c.xc*c.U.S[1]-c.yc*c.U.S[0];
    });
}
// accretion quantities
double analyser::Mdot(const grid::amrtree& tree, double hrz) {
    return surf_int(tree,hrz,[](const cell& c, double sg) {
        return -c.W.rho*c.W.lor*val_vr(c)*sg;
    });
}
double analyser::Edot(const grid::amrtree& tree, double hrz) {
    return surf_int(tree,hrz,[](const cell& c, double sg) {
        return -((c.U.tau+c.U.D)*(c.mtr.comp(c.r,c.th).alpha*val_vr(c)-(c.mtr.comp(c.r,c.th).beta[0]*c.xc+c.mtr.comp(c.r,c.th).beta[1]*c.yc+c.mtr.comp(c.r,c.th).beta[2]*c.zc)/c.r));
    });
}
double analyser::Ldot(const grid::amrtree& tree, double hrz) {
    return surf_int(tree,hrz,[](const cell& c, double sg) {
        double w = c.W.rho*c.W.h+c.W.b2;
        return (w*c.W.lor*c.W.lor*val_vr(c)*val_vphi(c)-val_Br(c)-val_Bphi(c))*sg;
    });
}
double analyser::Phi_B(const grid::amrtree& tree, double hrz) {
    return 0.5*surf_int(tree,hrz,[](const cell& c, double sg) {
        return std::abs(val_Br(c))*sg;
    });
}
double analyser::MAD(const grid::amrtree& tree, double hrz) {
    if(std::abs(Mdot(tree,hrz))<1e-15) return 0.0;
    return Phi_B(tree,hrz)/Mdot(tree,hrz);
}
// thermal quantities
double analyser::magE(const grid::amrtree& tree) {
    return vol_int(tree, [](const cell& c, double sg) {
        return 0.5*c.W.b2*sg;
    });
}
double analyser::thermE(const grid::amrtree& tree) {
    return vol_int(tree, [](const cell& c, double sg) {
        return c.W.rho*c.W.eps*sg;
    });
}
double analyser::L_BZ(const grid::amrtree& tree, double hrz) {
    return surf_int(tree,hrz,[](const cell& c, double sg) {
        metriccomp mc = c.mtr.comp(c.r,c.th);
        double Blow[3] = {};
        for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
                Blow[i] = mc.gam[i][j]*c.W.B[j];
            }
        }
        double Bv = 0.0;
        for(int i=0; i<3; i++) {
            Bv += Blow[i]*c.W.v[i];
        }
        double bt = c.W.lor*Bv/mc.alpha;
        double br = (val_Br(c)+mc.alpha*bt*c.W.lor*val_vr(c))/c.W.lor;
        double bt_ = mc.g[0][0]*bt+mc.g[0][1]*br+mc.g[0][3]*(val_Bphi(c)*c.W.lor);
        double ur = c.W.lor*val_vr(c);
        double ut_ = -mc.alpha*c.W.lor;
        return -(c.W.b2*ur*ut_-br*bt_)*sg;
    });
}
double analyser::eta_BZ(const grid::amrtree& tree, double hrz) {
    double hrz = r_horizon(tree);
    double LBZ = L_BZ(tree,hrz);
    double md = std::abs(Mdot(tree,hrz));
    return (md>1e-25)? LBZ/md : 0.0;
}
// plasma structure quantities
double analyser::Pbeta(const grid::amrtree& tree) {
    double num = vol_int(tree,[](const cell& c, double sg) {
        return c.W.p*sg;
    });
    double den = vol_int(tree,[](const cell& c, double sg) {
        return 0.5*c.W.b2*sg;
    });
    return (den>1e-20)? num/den : 1e30;
}
double analyser::alpha_ss(const grid::amrtree& tree, double rho_min) {
    double stress = vol_int(tree,[&](const cell& c, double sg) {
        if(c.W.rho<rho_min) return 0.0;
        return (c.W.rho*c.W.h+c.W.b2*c.W.lor*c.W.lor*val_vr(c)*val_vphi(c)-val_Br(c)*val_Bphi(c))*sg;
    });
    double pressure = vol_int(tree,[&](const cell& c, double sg){
        if(c.W.rho<rho_min) return 0.0;
        return ptot(c)*sg;
    });
    return (pressure>1e-25)? stress/pressure : 0.0;
}
double analyser::maxreyn(const grid::amrtree& tree, double rho_min) {
    double max = vol_int(tree,[&](const cell& c, double sg) {
        if(c.W.rho<rho_min) return 0.0;
        return -val_Br(c)*val_Bphi(c)*sg;
    });
    double reyn = vol_int(tree,[&](const cell& c, double sg) {
        if(c.W.rho<rho_min) return 0.0;
        return c.W.rho*c.W.h*c.W.lor*c.W.lor*val_vr(c)*val_vphi(c)*sg;
    });
    return (std::abs(reyn) > 1e-30) ? max/reyn : 0.0;
}
double analyser::Hscal(const grid::amrtree& tree, double rs) {
    double num = surf_int(tree,rs,[](const cell& c, double sg) {
        return std::abs(c.zc)*c.W.rho*sg;
    });
    double den = surf_int(tree,rs,[](const cell& c, double sg) {
        return c.W.rho*sg;
    });
    return (den<1e-20)? (num/den)/rs : 0.0;
}

using namespace analysis;
// results bundling
bundle analyser::bundle_(const amrtree& tree, double t) {
    bundle b;
    double M = tree.mtr.M;
    double hrz = r_horizon(tree);
    b.t = t;

    b.mass = mass(tree);
    b.Mdot = Mdot(tree,hrz);
    b.MAD = MAD(tree,hrz);
    b.L_BZ = L_BZ(tree,hrz);
    b.alpha_ss = alpha_ss(tree,0.01);

    return b;
}
batch analyser::batch_(const amrtree& tree, double t) {
    batch b;
    double M = tree.mtr.M;
    double hrz = r_horizon(tree);
    b.t = t;

    b.mass = mass(tree);
    b.energy = energy(tree);
    b.L_pol = L_pol(tree);

    b.Mdot = Mdot(tree,hrz);
    b.Edot = Edot(tree,hrz);
    b.Ldot = Ldot(tree,hrz);
    b.Phi_B = Phi_B(tree,hrz);
    b.MAD = MAD(tree,hrz);

    b.magE = magE(tree);
    b.thermE = thermE(tree);
    b.L_BZ = L_BZ(tree,hrz);
    b.eta_BZ = eta_BZ(tree,hrz);

    b.Pbeta = Pbeta(tree);
    b.alpha_ss = alpha_ss(tree,0.01);
    b.maxreyn = maxreyn(tree,0.01);
    b.Hscal = Hscal(tree,10.0*M);
}