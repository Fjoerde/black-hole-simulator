#include "metric.hpp"
#include <cmath>
#include <vector>
#include <array>
#include <iostream>

// this document contains the methods responsible for computing
// the various metric struct components, as well as the christoffel
// symbols and other quantities.

// compute \Sigma and H parameters in kerr-schild
double metric::Sigma(double r, double th) const {
    return r*r + a*a*pow(cos(th),2);
}
double metric::H(double r, double th) const {
    return (2*M*r - Q*Q)*pow(Sigma(r,th),-1);
}

// compute metric and inverse metric components
void metric::gcova(double r, double th, double g[4][4]) const {
    // first set all components to zero, then overwrite
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            g[i][j] = 0.0;
        }
    }
    // see documentation for reference
    g[0][0] = -1 + H(r,th);
    g[0][1] = g[1][0] = H(r,th);
    g[1][1] = 1 + H(r,th);
    g[2][2] = Sigma(r,th);
    g[0][3] = g[1][3] = g[3][0] = g[3][1] = -H(r,th)*a*pow(sin(th),2);
    g[3][3] = (r*r + a*a + H(r,th)*a*a*pow(sin(th),2))*pow(sin(th),2);
}
void metric::gcntr(double r, double th, double g_inv[4][4]) const {
    // same procedure as above; overwrite the zeros
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            g_inv[i][j] = 0.0;
        }
    }
    // see documentation for reference
    g_inv[0][0] = -1 - H(r,th);
    g_inv[0][1] = g_inv[1][0] = H(r,th);
    g_inv[1][1] = 1 - H(r,th);
    g_inv[2][2] = pow(Sigma(r,th),-1);
    g_inv[1][3] = g_inv[3][1] = a*pow(Sigma(r,th),-1);
    g_inv[3][3] = pow(Sigma(r,th)*sin(th),-1);
}

// compute metric determinant
void metric::gdet(double a, double r, double th, double sqrtdetg) const {
    sqrtdetg = (r*r+a*a*pow(sin(th),2))*sin(th);
}

// compute christoffel symbols
void metric::christ(double M, double a, double Q, double r, double th, double g_inv[4][4], double Gamma[4][4][4]) const {
    // first, define the required derivatives in the form
    // \partial_{\lambda}g_{\mu\nu} = dgdx[l][m][n], where the indices are
    // arbitrary and run between 0 and 3. there are 8 distinct terms.
    double dgdx[4][4][4];
    double metderiv(double M, double a, double Q, double r, double th, double dgdx[4][4][4]); {
        for(int i=0; i<4; i++) {
            for(int j=0; j<4; j++) {
                for(int k=0; k<4; k++) {
                    dgdx[i][j][k] = 0;
                }
            }
        }
    }

    // define god-awful expressions for metric derivatives (see documentation for reference)
    dgdx[1][0][0] = dgdx[1][0][1] = dgdx[1][1][0] = dgdx[1][1][1] = -2*(M*r*r-Q*Q*r-M*a*a*pow(cos(th),2))*pow(r*r+a*a*pow(cos(th),2),-2);
    dgdx[2][0][0] = dgdx[2][0][1] = dgdx[2][1][0] = dgdx[2][1][1] = 2*a*a*(2*M*r-Q*Q)*cos(th)*sin(th)*pow(r*r+a*a*pow(cos(th),2),-1);
    dgdx[1][2][2] = 2*r;
    dgdx[2][2][2] = -a*a*cos(th)*sin(th);
    dgdx[1][0][3] = dgdx[1][3][0] = dgdx[1][1][3] = dgdx[1][3][1] = 2*a*a*pow(sin(th),2)*(M*r*r-Q*Q*r-M*a*a*pow(cos(th),2))*pow(r*r+a*a*pow(cos(th),2),-2);
    dgdx[2][0][3] = dgdx[2][3][0] = dgdx[2][1][3] = dgdx[2][3][1] = 2*a*cos(th)*sin(th)*(2*M*r-Q*Q)*(r*r+a*a)*pow(r*r+a*a*pow(cos(th),2),-2);
    dgdx[1][3][3] = pow(sin(th),2)*(((2*M*a*a*pow(sin(th),2))*pow(r*r+a*a*pow(cos(th),2),-1))+((2*r*a*a*pow(sin(th),2)*(2*M*r-Q*Q))*pow(r*r+a*a*pow(cos(th),2),-2))+2*r);
    dgdx[2][3][3] = pow(sin(th),2)*(((2*pow(a,4)*(2*M*r-Q*Q)*cos(th)*pow(sin(th),3))*pow(r*r+a*a*pow(cos(th),2),-2))+((2*a*a*(2*M*r-Q*Q)*cos(th)*sin(th))*pow(r*r+a*a*pow(cos(th),2),-1)))+cos(th)*sin(th)*(((a*a*pow(sin(th),2)*(2*M*r-Q*Q))*pow(r*r+a*a*pow(cos(th),2),-1))+r*r+a*a);

    // combine according to the definition of the christoffel symbol to generate values.
    for(int l=0; l<4; l++) {
        for(int m=0; m<4; m++) {
            for(int n=0; n<4; n++) {
                Gamma[l][m][n] = 0.5*(g_inv[l][0]*(dgdx[m][0][n]+dgdx[n][0][m]-dgdx[0][m][n])+g_inv[l][1]*(dgdx[m][1][n]+dgdx[n][1][m]-dgdx[1][m][n])+g_inv[l][2]*(dgdx[m][2][n]+dgdx[n][2][m]-dgdx[2][m][n])+g_inv[l][3]*(dgdx[m][3][n]+dgdx[n][3][m]-dgdx[3][m][n]));
            }
        }
    }
}

// compute 3+1 ADM lapse and shift
void metric::lapse(double H, double r, double th, double alpha) const {
    alpha = pow(1+metric::H(r,th),-1/2);
}
void metric::shift(double g_inv[4][4], double r, double th, double beta[3]) const {
    for (int i=0; i<3; i++) {
        beta[i] = -g_inv[0][i+1]/g_inv[0][0];
    }
}

// compute spatial metric, inverse, and determinant
void metric::smgam(double g[4][4], double gam[3][3]) const {
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            gam[i][j] = g[i+1][j+1];
        }
    }
}
void metric::smgaminv(double g_inv[4][4], double gam_inv[3][3]) const {
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            gam_inv[i][j] = g_inv[i+1][j+1];
        }
    }
}
void metric::smgamdet(double alpha, double sqrtdetg, double sqrtdetgam) const {
    sqrtdetgam = alpha*sqrtdetg;
}

// assign expressions to metriccomp struct
metriccomp metric::comp(double r, double th) const {
    metriccomp mc;
    gcova(r,th,mc.g);
    gcntr(r,th,mc.g_inv);
    gdet(a,r,th,mc.sqrtdetg);
    christ(M,a,Q,r,th,mc.g_inv,mc.Gamma);
    lapse(H(r,th),r,th,mc.alpha);
    shift(mc.g_inv,r,th,mc.beta);
    smgam(mc.g, mc.gam);
    smgaminv(mc.g_inv,mc.gam_inv);
    smgamdet(mc.alpha,mc.sqrtdetg,mc.sqrtdetgam);
    return mc;
}