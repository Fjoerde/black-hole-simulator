#include <cmath>
#include <vector>
#include <array>
#include <memory>
#include "recon.hpp"
#include "grid.hpp"

// this document contains the methods for primitive left and right
// state reconstruction at cell faces.
using namespace grid;

// get index offsets
static const prim& getW(const patch& p, int i, int j, int k, int dim, int off) {
    int di = (dim==0)*off; int dj = (dim==1)*off; int dk = (dim==2)*off;
    return p.cell_(i+di,j+dj,k+dk).W;
}
// limiter choice switch
static double lim_(double dl, double dr, limiter lim) {
    switch(lim) {
        case limiter::minmod : return minmod(dl,dr);
        case limiter::vanleer : return vanleer(dl,dr);
        case limiter::mclim : return mclim(dl,dr);
        default : return minmod(dl,dr);
    }
}

// scalar reconstruction
static std::array<double,2> recon_scal(double Wm, double Wc, double Wp, double Wpp, limiter lim) {
    // left face
    double dLc = Wc-Wm; double dRc = Wp-Wc;
    double cslop = lim_(dLc,dRc,lim);
    double faceL = Wc+cslop/2.0;
    // right face
    double dLp = Wp-Wc; double dRp = Wpp-Wp;
    double pslop = lim_(dLp,dRp,lim);
    double faceR = Wp-pslop/2.0;
    return {faceL,faceR};
}
// reconstruction of face states of primitives
facestate reconfp(const patch& p, int i, int j, int k, int dim, limiter lim) {
    // gather extra cells for stencil
    const prim& Wm = getW(p,i,j,k,dim,-1);
    const prim& Wc = getW(p,i,j,k,dim,0);
    const prim& Wp = getW(p,i,j,k,dim,1);
    const prim& Wpp = getW(p,i,j,k,dim,2);
    facestate fs;

    // density, forcing positivity with ln(\rho) since \exp(ln\rho)>0
    double km = std::log(Wm.rho);
    double kc = std::log(Wc.rho);
    double kp = std::log(Wp.rho);
    double kpp = std::log(Wpp.rho);
    std::array<double,2> sLR = recon_scal(km,kc,kp,kpp,lim);
    fs.L.rho = std::exp(sLR[0]); fs.R.rho = std::exp(sLR[1]);
    // energy with the same logarithm-exponential positivity forcing
    double Em = std::log(Wm.eps);
    double Ec = std::log(Wc.eps);
    double Ep = std::log(Wp.eps);
    double Epp = std::log(Wpp.eps);
    std::array<double,2> JLR = recon_scal(Em,Ec,Ep,Epp,lim);
    fs.L.eps = std::exp(JLR[0]); fs.R.eps = std::exp(JLR[1]);

    // velocity, reconstructed directly
    for(int d=0; d<3; d++) {
        std::array<double,2> JRW = recon_scal(Wm.v[d],Wc.v[d],Wp.v[d],Wpp.v[d],lim);
        fs.L.v[d] = JRW[0]; fs.R.v[d] = JRW[1];
    }
    // causality check that v < c
    auto causalv = [](prim& W, const cell& c) {
        double v2 = 0.0;
        metriccomp mc = c.mtr.comp(c.r,c.th);
        for(int m=0; m<3; m++) {
            for(int n=0; n<3; n++) {
                v2 += mc.gam[m][n]*W.v[m]*W.v[n];
            }
        }
        // if v > c, scale to just barely be causal velocity
        if(v2>=1.0) {
            double s = (1-1e-10)/std::sqrt(v2);
            for(int a=0; a<3; a++) {W.v[a] *= s;};
        }
    };
    causalv(fs.L,p.cell_(i,j,k)); causalv(fs.R,p.cell_(i+(dim==0),j+(dim==1),k+(dim==2)));

    // magnetic field reconstructed to maintain zero divergence
    for(int d=0; d<3; d++) {
        if(d==dim) {
            // no normal component reconstruction; overwrite later
            fs.L.B[d] = Wc.B[d]; fs.R.B[d] = Wp.B[d];
        } else {
            std::array<double,2> JRE = recon_scal(Wm.B[d],Wc.B[d],Wp.B[d],Wpp.B[d],lim);
            fs.L.B[d] = JRE[0]; fs.R.B[d] = JRE[1];
        }
    }

    return fs;
}