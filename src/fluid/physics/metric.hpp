#pragma once
#include <cmath>
#include <vector>
#include <array>

// this document holds the metric class and components struct.
// it contains the basic function headers for metric components,
// as well as the Christoffel symbols, and other relevant
// quantities i.e. lapse, shift, and spatial metric.

struct metriccomp {
    double g[4][4]; // metric tensor
    double g_inv[4][4]; // inverse metric tensor
    double Gamma[4][4][4]; // christoffel symbols
    double sqrtdetg; // \sqrt{-g}
    double gam[3][3]; // spatial metric
    double gam_inv[3][3]; // inverse spatial metric
    double sqrtdetgam; // \sqrt{gamma}
    double alpha; // lapse
    double beta[3]; // shift
};

class metric {
public:
    // black hole parameters
    double M; // mass
    double a; // spin
    double Q; // charge

    metric(double M, double a, double Q): M(M), a(a), Q(Q) {};

    metriccomp comp(double r, double th); // compute metric at given coordinates, only requiring r and \theta

    // metric component functions
    void gcova(double r, double th, double g[4][4]) const;
    void gcntr(double r, double th, double g_inv[4][4]) const;
    void christ(double M, double a, double Q, double r, double th, double g_inv[4][4], double Gamma[4][4][4]) const;
    void gdet(double a, double r, double th, double sqrtdetg) const;

    // lapse and shift functions
    void lapse(double H, double r, double th, double alpha) const;
    void shift(double g_inv[4][4], double r, double th, double beta[3]) const;

    // spatial metric component functions
    void smgam(double g[4][4], double gam[3][3]) const;
    void smgaminv(double g_inv[4][4], double gam_inv[3][3]) const;
    void smgamdet(double alpha, double sqrtdetg, double sqrtdetgam) const;

    private:
    double Sigma(double r, double th) const;
    double H(double r, double th) const;
};
