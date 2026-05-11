#pragma once
#include <cmath>
#include <vector>
#include <array>
#include "metric.hpp"
#include "state.hpp"
#include "prim.hpp"
#include "cons.hpp"

// this document contains the structures regarding the grid cells.

// struct for each cell
struct cell {
    // index
    int i, j, k; // spatial index within patch
    // variables
    cons U;
    prim W;
    cons dU; // conserved variable accumulator for RK2
    // amr details + flags
    int lvl;
    bool ghost;
    bool refine;
    bool erase; 
    // spacetime
    metric mtr; // metric
    double r; double th; double phi; // kerr-schild coordinates of cell centre
    double xc; double yc; double zc; // cartesian coordinates of cell centre
    double dx; double dy; double dz; // cell side lengths in (x,y,z)
    double vol; // cell coordinate volume
    double pvol; // proper volume pvol = vol/mtr.sqrtdetg
};