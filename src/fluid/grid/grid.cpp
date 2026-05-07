#include <cmath>
#include <vector>
#include <array>
#include <memory>
#include "grid.hpp"
#include "cell.hpp"
#include "metric.hpp"

// this document contains the methods for dealing with the grid and
// adaptive mesh refinement (amr). the grid is split up into a quilt
// of patches, with each patch containing cells which hold data.

// patch setup
patch::patch(int lvl, std::array<int,3> ijk, std::array<double,3> cor1, std::array<double,3> cor2, patch* parent) : lvl(lvl), ijk(ijk), parent(parent), children({nullptr}), leaf(true) {
    // active cell edge arrays
    xedge.resize(block+1);
    yedge.resize(block+1);
    zedge.resize(block+1);
    for(int i=0; i<=block; i++) {
        double k = double(i)/block;
        xedge[i] = cor1[0]+k*(cor2[0]-cor1[0]);
        yedge[i] = cor1[1]+k*(cor2[1]-cor1[1]);
        zedge[i] = cor1[2]+k*(cor2[2]-cor1[2]);
    }
    // cells and fluxes
    cells.resize(cpn*cpn*cpn);
    Fx.resize((cpn+1)*cpn*cpn);
    Fy.resize(cpn*(cpn+1)*cpn);
    Fz.resize(cpn*cpn*(cpn+1));

    cell_init();
}

// patch and cell indexing
int patch::lindx(int i, int j, int k) const {
    return (i+ghost)*cpn*cpn+(j+ghost)*cpn+k+ghost;
}
cell& patch::cell_(int i, int j, int k) {
    return cells[lindx(i,j,k)];
}
const cell& patch::cell_(int i, int j, int k) const {
    return cells[lindx(i,j,k)];
}

// geometry
std::array<double,3> patch::cellmid(int i, int j, int k) const {
    return {(xedge[i]+xedge[i+1])/2, (yedge[j]+yedge[j+1])/2, (zedge[k]+zedge[k+1])/2};
}
std::array<double,3> patch::corner(int i, int j, int k) const {
    return {xedge[i], yedge[j], zedge[k]};
}
// edge lengths
double patch::dx() const {return xedge[1]-xedge[0];}
double patch::dy() const {return yedge[1]-yedge[0];}
double patch::dz() const {return zedge[1]-zedge[0];}

// cell initialisation
void patch::cell_init() {
    for(int i=-ghost; i<block+ghost; i++) {
        for(int j=-ghost; j<block+ghost; j++) {
            for(int k=-ghost; k<block+ghost; k++) {
                // geometry
                cell& c = cell_(i,j,k);
                c.i = i; c.j = j; c.k = k;
                c.dx = dx(); c.dy = dy(); c.dz = dz();
                c.xc = xedge[0]+(i+0.5)*dx(); c.yc = yedge[0]+(j+0.5)*dy(); c.zc = zedge[0]+(k+0.5)*dz();
                c.vol = c.dx*c.dy*c.dz;
                // amr information
                c.lvl = lvl;
                c.refine = false;
                c.erase = false;
                c.ghost = (i<0 || i>=block || j<0 || j>=block || k<0 || k>=block);
                // spacetime
                c.r = std::sqrt(c.xc*c.xc+c.yc*c.yc+c.zc*c.zc);
                c.th = std::acos(c.zc/c.r);
                c.phi = std::atan2(c.yc,c.xc);
                metriccomp mc = c.mtr.comp(c.r,c.th);
                c.mtr.gcova(c.r,c.th,mc.g);
                c.pvol = c.vol/mc.sqrtdetg;
            }
        }
    }
}
// amr tree initialisation
amrtree::amrtree(std::array<double,3> dom_l, std::array<double,3> dom_h, int nqlt) {
    // coordinate sizes of each initial patch
    double px = (dom_h[0]-dom_l[0])/nqlt; double py = (dom_h[1]-dom_l[1])/nqlt; double pz = (dom_h[2]-dom_l[2])/nqlt;
    // divide quilt into nqlt x nqlt x nqlt patches + index in quilt
    for(int i=0; i<nqlt; i++) {
        for(int j=0; j<nqlt; j++) {
            for(int k=0; k<nqlt; k++) {
                std::array<double,3> c1 = {dom_l[0]+i*px, dom_l[1]+j*py, dom_l[2]+k*pz};
                std::array<double,3> c2 = {c1[0]+px, c1[1]+py, c1[2]+pz};
                quilt.push_back(std::make_unique<patch>(0, std::array<int,3>{i,j,k}, c1, c2, nullptr));
            }
        }
    }
}