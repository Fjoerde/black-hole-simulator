#include <cmath>
#include <vector>
#include <array>

#include "hlld.hpp"
#include "grid.hpp"

// this document contains the methods for the HLLD solver.
using namespace grid;

// internal structs
// precompute values at the faces and cache
struct side {
    double rho, eps, p, h; // thermodynamic quantities
    double vn, vt1, vt2; // normal and transverse velocities
    double Bn, Bt1, Bt2; // transverse and normal magnetic field
    double v2; // square norm of velocity
    double ltz, ltz2; // lorentz
    double b2; // square norm of lagrangian magnetic 4-vector
    double pt; // total pressure
    double cf; // fast magnetosonic speed
    double D, tau; // conserved D and \tau
    double Sn, St1, St2; // conserved normal and transverse momentum
    double Bc; // conserved eulerian magnetic field
    double E; // total energy
};
// conserved variable intermediate state
struct cons_stt {
    double D, Sn, St1, St2, tau, Bn, Bt1, Bt2;
};

// helper functions
// fast magnetosonic wave speeds
static double fastspeed(double rho, double h, double b2, double cs2) {
    double va2 = b2/(rho*h+b2); // square of alfvén wave speed
    double cf2 = cs2+va2*(1-cs2); // angle-averaged fast waves peed
    cf2 = std::min(cf2, 1-1e-10);
    return std::sqrt(cf2);
}
// magnetic related computations
static void magcomp(const prim& W, const metriccomp& mc, int dim, double& b2, double& pt, double& Bn, double& Bt1, double& Bt2, double& vn, double& vt1, double& vt2, double& v2, double& ltz, double& ltz2) {
    int t1 = (dim+1)%3, t2 = (dim+2)%3;
    // velocity and lorentz factor
    v2 = 0.0;
    double vlow[3] = {}, Blow[3] = {};
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            vlow[i] += mc.gam[i][j]*W.v[j];
            Blow[i] += mc.gam[i][j]*W.B[j];
        }
        v2 += vlow[i]*W.v[i];
    }
    v2 = std::min(v2,1-1e-10);
    ltz2 = 1.0/(1.0-v2);
    ltz = std::sqrt(ltz2);
    // B_i v^i
    double Bv = 0.0;
    for(int i=0; i<3; i++) {
        Bv += Blow[i]*W.v[i];
    }
    // B_i B^i
    double Bsq = 0.0;
    for(int i=0; i<3; i++) {
        Bsq += Blow[i]*W.B[i];
    }
    // value assignments
    b2 = Bv*Bv+Bsq/ltz2;
    pt = 0.0+b2/2;
    vn = W.v[dim]; vt1 = W.v[t1]; vt2 = W.v[t2];
    Bn = W.B[dim]; Bt1 = W.B[t1]; Bt2 = W.B[t2];
}
// side initialisation
static side side_(const prim& W, const cons& U, const metriccomp& mc, const state& stt, double sqrtg, int dim) {
    side s;
    int t1 = (dim+1)%3; int t2 = (dim+2)%3;
    // assign values
    s.rho = W.rho; s.eps = W.eps;
    s.p = stt.press(W.rho,W.eps);
    s.h = stt.enth(W.rho,W.eps);
    // pressure
    double dum_pt;
    magcomp(W,mc,dim,s.b2,dum_pt,s.Bn,s.Bt1,s.Bt2,s.vn,s.vt1,s.vt2,s.v2,s.ltz,s.ltz2);
    s.pt = s.p+s.b2/2;
    // sound speeds
    double cs2 = stt.cs2(W.rho,W.eps);
    s.cf = fastspeed(s.rho,s.h,s.b2,cs2);
    // conserved stripping
    s.D = U.D/sqrtg; s.tau = U.tau/sqrtg;
    s.Sn = U.S[dim]/sqrtg; s.St1 = U.S[t1]/sqrtg; s.St2 = U.S[t2]/sqrtg;
    s.Bc = U.B[dim]/sqrtg;
    s.E = s.tau+s.D;

    return s;
}

// physical flux
static cons_stt physflux(const side& s, const metriccomp& mc, int dim) {
    cons_stt F;
    double vn = s.vn;
    // fluxes in each variable
    F.D = s.D*vn;
    F.tau = (s.E+s.pt)*vn-s.Bn*(s.Bn*vn+s.Bt1*s.vt1+s.Bt2*s.vt2)/s.ltz2;
    F.Sn = s.Sn*vn-s.Bn*s.Bn/s.ltz2+s.pt;
    F.St1 = s.St1*vn-s.Bt1*s.Bn/s.ltz2;
    F.St2 = s.St2*vn-s.Bt2*s.Bn/s.ltz2;
    F.Bn = 0.0;
    F.Bt1 = s.Bt1*vn-s.Bn*s.vt1;
    F.Bt2 = s.Bt2*vn-s.Bn*s.vt2;
    // ADM 3+1 correction for near-horizon fluxes
    double a = mc.alpha;
    double btn = mc.beta[dim];
    F.D = a*F.D-btn*s.D;
    F.tau = a*F.tau-btn*s.tau;
    F.Sn = a*F.Sn-btn*s.Sn;
    F.St1 = a*F.St1-btn*s.St1;
    F.St2 = a*F.St2-btn*s.St2;
    F.Bn = a*F.Bn-btn*s.Bn;
    F.Bt1 = a*F.Bt1-btn*s.Bt1;
    F.Bt2 = a*F.Bt2-btn*s.Bt2;

    return F;
}
// hll flux fallback
static cons_stt hll_flux(const side& L, const side& R, const cons_stt& FL, const cons_stt& FR, double SL, double SR) {
    double dS = SR-SL; cons_stt F;
    // hll fluxes by variable; see toro ch10.3 for reference
    F.D = (SR*FL.D-SL*FR.D+SL*SR*(R.D-L.D))/dS;
    F.tau = (SR*FL.tau-SL*FR.tau+SL*SR*(R.tau-L.tau))/dS;
    F.Sn = (SR*FL.Sn-SL*FR.Sn+SL*SR*(R.Sn-L.Sn))/dS;
    F.St1 = (SR*FL.St1-SL*FR.St1+SL*SR*(R.St1-L.St1))/dS;
    F.St2 = (SR*FL.St2-SL*FR.St2+SL*SR*(R.St2-L.St2))/dS;
    F.Bn = (SR*FL.Bn-SL*FR.Bn+SL*SR*(R.Bn-L.Bn))/dS;
    F.Bt1 = (SR*FL.Bt1-SL*FR.Bt1+SL*SR*(R.Bt1-L.Bt1))/dS;
    F.Bt2 = (SR*FL.Bt2-SL*FR.Bt2+SL*SR*(R.Bt2-L.Bt2))/dS;

    return F;
}

// star state construction across fast waves U^*
static cons_stt star_stt(const side& s, cons_stt& F, double SK, double SM, double pt_x) {
    cons_stt Ux;
    Ux.D = s.D*(SK-s.vn)/(SK-SM);
    Ux.tau = (s.E*(SK-s.vn)-s.pt*s.vn+pt_x*SM)/(SK-SM)-Ux.D;
    Ux.Sn = (s.Sn*(SK-s.vn)-s.pt+pt_x)/(SK-SM);
    Ux.St1 = s.St1*(SK-s.vn)/(SK-SM);
    Ux.St2 = s.St2*(SK-s.vn)/(SK-SM);
    Ux.Bn = s.Bc;
    // transverse magnetic fields
    double den = s.D*(SK-s.vn)*(SK-SM)-s.Bn*s.Bn;
    if(std::abs(den)>1e-14) {
        double k = s.Bn*(SM-s.vn)/den;
        Ux.Bt1 = s.Bt1-s.Bn*(s.vt1*(SK-s.vn)+s.Bn*s.Bt1/s.D)*k;
        Ux.Bt2 = s.Bt2-s.Bn*(s.vt2*(SK-s.vn)+s.Bn*s.Bt2/s.D)*k;
    } else {
        Ux.Bt1 = s.Bt1; Ux.Bt2 = s.Bt2;
    }

    return Ux;
}
// double star state construction across alfvén waves U^{**}
static void alfstar_stt(const cons_stt& UxL, const cons_stt& UxR, const side& L, const side& R, double SL, double SR, double SM, double pt_x, cons_stt& UxxL, cons_stt& UxxR) {
    // preliminary U^{**} values preserved in D^* and \tau^*
    UxxL.D = UxL.D; UxxL.tau = UxL.tau;
    UxxR.D = UxR.D; UxxR.tau = UxR.tau;
    UxxL.Bn = UxL.Bn; UxxR.Bn = UxR.Bn;
    // alfvén wave speeds
    double sq_DL = std::sqrt(std::abs(UxL.D));
    double sq_DR = std::sqrt(std::abs(UxR.D));
    double Bn = L.Bn;
    double den = sq_DL+sq_DR;
    if(std::abs(den)<1e-14 || std::abs(Bn)<1e-14) {
        UxxL = UxL; UxxR = UxR;
    }

    double sgnBn = (Bn>=0.0)? 1.0 : -1.0;
    // transverse velocity in ** region
    double vt1xx = (sq_DL*UxL.St1/UxL.D+sq_DR*UxR.St1/UxR.D+sgnBn*(UxR.Bt1-UxL.Bt1))/den;
    double vt2xx = (sq_DL*UxL.St2/UxL.D+sq_DR*UxR.St2/UxR.D+sgnBn*(UxR.Bt2-UxL.Bt2))/den;
    // transverse magnetic field
    double Bt1xx = (sq_DL*UxL.Bt1/UxL.D+sq_DR*UxR.Bt1/UxR.D+sgnBn*sq_DL*sq_DR*(UxR.St1/UxR.D-UxL.St1/UxL.D))/den;
    double Bt2xx = (sq_DL*UxL.Bt2/UxL.D+sq_DR*UxR.Bt2/UxR.D+sgnBn*sq_DL*sq_DR*(UxR.St2/UxR.D-UxL.St2/UxL.D))/den;
    
    // assign states
    UxxL.St1 = UxL.D*vt1xx; UxxR.St1 = UxR.D*vt1xx;
    UxxL.St2 = UxL.D*vt2xx; UxxR.St2 = UxR.D*vt2xx;
    UxxL.Bt1 = Bt1xx; UxxR.Bt1 = Bt1xx;
    UxxL.Bt2 = Bt2xx; UxxR.Bt2 = Bt2xx;
    UxxL.Sn = UxL.D*SM; UxxR.Sn = UxR.D*SM;
    const double vt1xL = UxL.St1 / UxL.D;
    const double vt2xL = UxL.St2 / UxL.D;
    const double vt1xR = UxR.St1 / UxR.D;
    const double vt2xR = UxR.St2 / UxR.D;
    UxxL.tau = UxL.tau-sq_DL*sgnBn*(vt1xx*Bt1xx+vt2xx*Bt2xx-vt1xL*UxL.Bt1-vt2xL*UxL.Bt2);
    UxxR.tau = UxR.tau+sq_DR*sgnBn*(vt1xx*Bt1xx+vt2xx*Bt2xx-vt1xR*UxR.Bt1-vt2xR*UxR.Bt2);
    // UxxL.tau = UxL.tau-sq_DL*sgnBn*(UxL.St1/(UxL.D*UxL.Bt1)+UxL.St2/(UxL.D*UxL.Bt2)-vt1xx*Bt1xx-vt2xx*Bt2xx)-UxL.D;
    // UxxR.tau = UxR.tau-sq_DR*sgnBn*(UxR.St1/(UxR.D*UxR.Bt1)+UxR.St2/(UxR.D*UxR.Bt2)-vt1xx*Bt1xx-vt2xx*Bt2xx)-UxR.D;
}
// intermediate state flux
static cons_stt interflux(const cons_stt& F_K, double SK, const cons_stt& Ux, const side& s_K) {
    cons_stt F;
    // assign values
    F.D = F_K.D+SK*(Ux.D-s_K.D);
    F.tau = F_K.tau+SK*(Ux.tau-(s_K.E-s_K.D));
    F.Sn = F_K.Sn+SK*(Ux.Sn-s_K.Sn);
    F.St1 = F_K.St1+SK*(Ux.St1-s_K.St1);
    F.St2 = F_K.St2+SK*(Ux.St2-s_K.St2);
    F.Bn = F_K.Bn+SK*(Ux.Bn-s_K.Bc);
    F.Bt1 = F_K.Bt1+SK*(Ux.Bt1-s_K.Bt1);
    F.Bt2 = F_K.Bt2+SK*(Ux.Bt2-s_K.Bt2);

    return F;
}
// restore flux with \sqrt{-g} factor proper ordering
static cons flux(const cons_stt& F, double sg, int dim) {
    int t1 = (dim+1)%3; int t2 = (dim+2)%3;
    cons c;
    // assign values
    c.D = sg*F.D; c.tau = sg*F.tau;
    c.B[dim] = sg*F.Bn; c.B[t1] = sg*F.Bt1; c.B[t2] = sg*F.Bt2;
    c.S[dim] = sg*F.Sn; c.S[t1] = sg*F.St1; c.S[t2] = sg*F.St2;
    
    return c;
}

// main hlld solver function
hlldflux hlld(const prim& WL, const prim& WR, const cons& UL, const cons& UR, const metric& mtr, double r, double th, const state& stt, int dim) {
    hlldflux fx;
    // metric
    metriccomp mc = mtr.comp(r,th);
    double sg = mc.sqrtdetg;
    // side structs
    side L = side_(WL,UL,mc,stt,sg,dim);
    side R = side_(WR,UR,mc,stt,sg,dim);

    // wave speeds estimates and fluxes
    double SL = std::min(mc.alpha*(L.vn-L.cf)-mc.beta[dim],mc.alpha*(R.vn-R.cf)-mc.beta[dim]);
    double SR = std::max(mc.alpha*(L.vn+L.cf)-mc.beta[dim],mc.alpha*(R.vn+R.cf)-mc.beta[dim]);
    cons_stt FL = physflux(L,mc,dim); cons_stt FR = physflux(R,mc,dim);
    fx.ctspeed = std::max(std::abs(SL),std::abs(SR));
    if(SR-SL<1e-12) {
        fx.F = flux(hll_flux(L,R,FL,FR,SL,SR),sg,dim);
    }
    if(SL>=0.0) {
        fx.F = flux(FL,sg,dim);
        return fx;
    }
    if(SR<=0.0) {
        fx.F = flux(FR,sg,dim);
        return fx;
    }
    // total pressure and contact waves
    double pt_x = (SR*L.pt-SL*R.pt+SL*SR*(R.Sn-L.Sn))/(SR-SL);
    double SM_u = SR*L.E-SL*R.E-FR.tau-FR.D+FL.tau+FL.D;
    double SM_d = SR*L.D-SL*R.D-FR.D+FL.D;
    if(std::abs(SM_d)<1e-12) {
        fx.F = flux(hll_flux(L,R,FL,FR,SL,SR),sg,dim);
        return fx;
    }
    double SM = SM_u/SM_d;
    // alfvén waves
    cons_stt UxL = star_stt(L,FL,SL,SM,pt_x);
    cons_stt UxR = star_stt(R,FR,SR,SM,pt_x);
    double SLx = SM-std::abs(L.Bn)/std::sqrt(std::abs(UxL.D));
    double SRx = SM+std::abs(R.Bn)/std::sqrt(std::abs(UxR.D));
    if(UxL.D<=0.0 || UxR.D<=0.0) {
        fx.F = flux(hll_flux(L,R,FL,FR,SL,SR),sg,dim);
        return fx;
    }

    // flux selector
    if(SM>=0.0) {
        if(SLx>=0.0) {
            // L^*
            cons_stt Fx = interflux(FL,SL,UxL,L);
            fx.F = flux(Fx,sg,dim);
        } else {
            // L^{**}
            cons_stt UxxL, UxxR;
            alfstar_stt(UxL,UxR,L,R,SL,SR,SM,pt_x,UxxL,UxxR);
            cons_stt FLx = interflux(FL,SL,UxL,L);
            cons_stt Fxx = interflux(FLx,SLx,UxxL,L);
            fx.F = flux(Fxx,sg,dim);
        }
    } else {
        if(SRx>=0.0) {
            // L^*
            cons_stt Fx = interflux(FR,SR,UxR,R);
            fx.F = flux(Fx,sg,dim);
        } else {
            // L^{**}
            cons_stt UxxL, UxxR;
            alfstar_stt(UxL,UxR,L,R,SL,SR,SM,pt_x,UxxL,UxxR);
            cons_stt FRx = interflux(FR,SR,UxR,R);
            cons_stt Fxx = interflux(FRx,SRx,UxxR,R);
            fx.F = flux(Fxx,sg,dim);
        }
    }
    // safety checks in case NaN/infinities/other errors slip through
    if(!std::isfinite(fx.F.D) || !std::isfinite(fx.F.tau) || !std::isfinite(fx.F.S[dim%3]) || !std::isfinite(fx.F.S[(dim+1)%3]) || !std::isfinite(fx.F.S[(dim+2)%3]) || !std::isfinite(fx.F.B[dim%3]) || !std::isfinite(fx.F.B[(dim+1)%3]) || !std::isfinite(fx.F.B[(dim+2)%3])) {
        fx.F = flux(hll_flux(L,R,FL,FR,SL,SR),sg,dim);
    }
    if(fx.ctspeed>1.0) {
        fx.F = flux(hll_flux(L,R,FL,FR,SL,SR),sg,dim);
        fx.ctspeed = std::min(fx.ctspeed,1.0);
    }

    return fx;
}