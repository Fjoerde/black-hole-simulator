#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <stdexcept>

#include "src/fluid/cons.hpp"
#include "src/fluid/metric.hpp"
#include "src/fluid/prim.hpp"

// this document contains the methods for analytical conserved
// variable construction and primitive variable reconstruction by
// root-finding iteration.

// primitive to conserved variable construction
// this process is analytical so is nice and easy in principle
cons conserved::ptoc(prim& pv, double r, double th) const {
    // metric setup
    metriccomp mc = mtr.comp(r, th);
    double sg = mc.sqrtdetgam;
    // covariant velocity
    double vlow[3] = {0.0, 0.0, 0.0};
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            vlow[i] += mc.gam[i][j]*pv.v[j];
        }
    }
    // covariant magnetic flux
    double Blow[3] = {0.0, 0.0, 0.0};
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            Blow[i] += mc.gam[i][j]*pv.B[j];
        }
    }
    // dot product \mathbf{B}\cdot\mathbf{v}
    double Bv = 0.0;
    for(int i=0; i<3; i++) {
        Bv += Blow[i]*pv.v[i];
    }
    
    double bt = pv.lor*Bv/mc.alpha; // magnetic 4-vector t component
    double pstar = pv.p+pv.b2/2; // magnetically adjusted pressure
    double w = pv.rho*pv.h+pv.b2; // magnetically adjusted enthalpy density

    // conserved variable construction and collection
    cons cv;
    cv.D = sg*pv.rho*pv.lor; // conserved density
    cv.tau = sg*(w*pv.lor*pv.lor-pstar-bt*bt)-cv.D; // conserved energy
    for(int i=0; i<3; i++) {
        double bi = (Blow[i]+mc.alpha*bt*pv.lor*vlow[i])/pv.lor;
        cv.S[i] = sg*(w*pv.lor*pv.lor*vlow[i]-mc.alpha*bt*bi); // conserved momentum
        cv.B[i] = sg*pv.B[i]; // conserved magnetic field components
    }
    return cv;
}

// there exist no closed-form inverses to recover primitive variables
// from the conserved ones, so an iterative root-finding solver must
// be used which behaves well everywhere, including near the horizon.

// define solver residuals and derivatives
double conserved::f_df(double xi, double Ssq, double Bsq, double SB, double Dn, double taun, double& df) const {
    // closed-form expressions are faster than numerical derivatives,
    // so define intermediate variables to mitigate the awful length
    // of parts of the expressions.
    double Yi = SB*SB;
    double K = Bsq;
    double X = xi*xi;
    double xipK = xi+K;
    double vsq = (Ssq*X+Yi*(2.0*xi+K))/(X+xipK*xipK);
    // causality condition
    if(vsq>=1.0) {
        df = 0.0;
        return 1e30;
    }
    // lorentz factor and thermodynamic quantities
    double ltz2 = 1.0/(1.0-vsq);
    double ltz = std::sqrt(ltz2);
    double rho = Dn/ltz;
    double G = stt.gamma-1.0;
    double p = (G/stt.gamma)*(xi/ltz2-Dn/ltz);
    double pstar = p+K/(2*ltz2)+Yi/(2*X); // total effective pressure
    // physicality condition
    if(p<0.0 || rho<0.0) {
        df = 0.0;
        return 1e32;
    }
    // energy residual
    double f = xi-taun-Dn-pstar;

    // define intermediate quantities for closed-form derivatives
    // use quotient rule with |\mathbf{v}|^2=\frac{N}{M}
    double N = Ssq*X+Yi*(2.0*xi+K);
    double M = X*xipK*xipK;
    double dN = 2.0*Ssq*xi+2.0*Yi;
    double dM = 2.0*xi*xipK*xipK+2.0*X*X;
    double dvv = (dN*M-dM*N)/(M*M); // \frac{d|\mathbf{v}|^2}{d\xi}
    // pressure derivatives
    double dltz2 = ltz2*ltz2*dvv;
    double dltz = dltz2/(2.0*ltz);
    double dp = (G/stt.gamma)*((1.0/ltz2)-(xi*dltz2)/(ltz2*ltz2)+Dn*dltz/ltz2);
    double dpstar = dp-(K*dltz2)/(2.0*ltz2*ltz2)-Yi/(xi*X);
    // return quantities
    df = 1-dpstar;
    return f;
}
double conserved::fres(double xi, double Ssq, double Bsq, double SB, double Dn, double taun) const {
    double dumdf;
    return conserved::f_df(xi, Ssq, Bsq, SB, Dn, taun, dumdf);
}

// for reconstruction, first attempt newton-raphson iteration, and if
// it fails within the first few attempts (such as near the horizon),
// switch to a more robust solver.

// conserved to primitive variable reconstruction
bool conserved::ctop(const cons& cv, double r, double th, prim& pv_out, int maxiter=50, double tol=1e-8) const {
    // set up variables
    metriccomp mc = mtr.comp(r, th);
    double sg = mc.sqrtdetgam;
    double D = cv.D/sg;
    double tau = cv.tau/sg;
    double B[3], S[3];
    double Bsq = 0.0; double Ssq = 0.0; double SB = 0.0;
    double Bu[3], Su[3];
    for(int i=0; i<3; i++) {
        B[i] = cv.B[i]/sg;
        S[i] = cv.S[i]/sg;
        for(int j=0; j<3; j++) {
            Bu[i] = mc.gam_inv[i][j]*B[j];
            Su[i] = mc.gam_inv[i][j]*S[j];
        }
        Ssq += S[i]*Su[i];
        Bsq += B[i]*Bu[i];
        SB += S[i]*Bu[i];
    }
    double xi = tau+D;

    // bracket for lower and upper bounds on \xi
    double xi_l = D;
    double xi_h = tau+D+Bsq/2+1e3;
    // check that bracket is valid
    double fl = fres(xi_l, Ssq, Bsq, SB, D, tau);
    double fh = fres(xi_h, Ssq, Bsq, SB, D, tau);
    // expand bracket while fl and fh are the same sign
    while(fl*fh>0.0 && xi_h < 1e8*D) {
        xi_h *= 2.0;
        fl = fres(fl, Ssq, Bsq, SB, D, tau);
    }
    
    // try newton-raphson first
    double xi = (xi_l+xi_h)/2.0;
    bool nr_ok_la = true;
    // iterate
    for(int i=0; i<maxiter; i++) {
        double dF;
        double F = f_df(xi, Ssq, Bsq, SB, D, tau, dF);
        if(std::abs(dF)<1e-30) {
            nr_ok_la = false;
            break;
        }
        double xin = xi-F/dF;
        if(xin<xi_l || xin>xi_h) {
            nr_ok_la = false;
            break;
        }
        xi = xin;
    }
    // use brent if newton-raphson fails
    if(!nr_ok_la) {
        double xib;

    }
    return true;
}
// brent-dekker solver
bool conserved::cp_bd(double xa, double xb, double Ssq, double Bsq, double SB, double D, double tau, double& root, double tol=1e-12) const {
    double fa = conserved::fres(xa, Ssq, Bsq, SB, D, tau);
    double fb = conserved::fres(xb, Ssq, Bsq, SB, D, tau);
    // impose dekker conditions
    if(fa*fb>0.0) {
        return false;
    }
    if(std::abs(fa)<std::abs(fb)) {
        std::swap(xa,xb);
        std::swap(fa,fb);
    }
    // more setup
    double xc = xa; double fc = fa;
    bool mflag = true;
    double s = 0.0; double d = 0.0;

    // iteration
    for(int i=0; i<100; i++) {
        if(std::abs(xb-xa)<tol) {
            root = xb;
            return true;
        }
        // inverse quadratic, otherwise secant
        if(fa!=fc && fb!=fc) {
            s = (xa*fb*fc)/((fa-fb)-(fa-fc))+(xb*fa*fc)/((fb-fa)-(fb-fc))+(xc*fa*fb)/((fc-fa)-(fc-fb));
        } else {
            s = xb-fb*(xb-xa)/(fb-fa);
        }
        // bisection conditions
        bool c1 = !((3*xa+4)/4<s && s<xb);
        bool c2 = mflag && std::abs(s-xb)>=std::abs(xb-xc)/2;
        bool c3 = !mflag && std::abs(s-xb)>=std::abs(xc-d)/2;
        // bisection
        if(c1 || c2 || c3) {
            s = (xa+xb)/2;
            mflag = true;
        } else {
            mflag = false;
        }
        // residuals
        double fs = conserved::fres(s, Ssq, Bsq, SB, D, tau);
        d = xc; xc = xb; fc = fb;
        if(fa*fs<0.0) {
            xb = s;
        } else {
            xa = s;
        }
        if(std::abs(fa)<std::abs(fb)) {
            std::swap(xa,xb);
            std::swap(fa,fb);
        }
    }
    return false;
}