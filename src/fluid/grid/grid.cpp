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
    
    // magnetic field and emf
    int g = ghost;
    Bfx.resize((block+2*g+1)*(block+2*g)*(block+2*g),0.0);
    Bfy.resize((block+2*g)*(block+2*g+1)*(block+2*g),0.0);
    Bfz.resize((block+2*g)*(block+2*g)*(block+2*g+1),0.0);
    EMFx.resize((block+2*g)*(block+2*g+1)*(block+2*g+1),0.0);
    EMFy.resize((block+2*g+1)*(block+2*g)*(block+2*g+1),0.0);
    EMFz.resize((block+2*g+1)*(block+2*g+1)*(block+2*g),0.0);

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
// face indexing
int patch::xfc_idx(int i, int j, int k) const {
    return (i+ghost)*(block+2*ghost)*(block+2*ghost)+(j+ghost)*(block+2*ghost)+(k+ghost);
}
int patch::yfc_idx(int i, int j, int k) const {
    return (i+ghost)*(block+2*ghost+1)*(block+2*ghost)+(j+ghost)*(block+2*ghost)+(k+ghost);
}
int patch::zfc_idx(int i, int j, int k) const {
    return (i+ghost)*(block+2*ghost)*(block+2*ghost+1)+(j+ghost)*(block+2*ghost+1)+(k+ghost);
}

// electromagnetic indexing
int patch::Bfx_idx(int i, int j, int k) const {
    return (i+ghost)*(block+2*ghost)*(block+2*ghost)+(j+ghost)*(block+2*ghost)+(k+ghost);
}
int patch::Bfy_idx(int i, int j, int k) const {
    return (i+ghost)*(block+2*ghost+1)*(block+2*ghost)+(j+ghost)*(block+2*ghost)+(k+ghost);
}
int patch::Bfz_idx(int i, int j, int k) const {
    return (i+ghost)*(block+2*ghost)*(block+2*ghost+1)+(j+ghost)*(block+2*ghost+1)+(k+ghost);
}
int patch::EMFx_idx(int i, int j, int k) const {
    return (i+ghost)*(block+2*ghost+1)*(block+2*ghost+1)+(j+ghost)*(block+2*ghost+1)+(k+ghost);
}
int patch::EMFy_idx(int i, int j, int k) const {
    return (i+ghost)*(block+2*ghost)*(block+2*ghost+1)+(j+ghost)*(block+2*ghost+1)+(k+ghost);
}
int patch::EMFz_idx(int i, int j, int k) const {
    return (i+ghost)*(block+2*ghost+1)*(block+2*ghost)+(j+ghost)*(block+2*ghost)+(k+ghost);
}
// magnetic flux accessors
double patch::get_Fx_B(int d, int i, int j, int k) const {
    return Fx[xfc_idx(i,j,k)].B[d];
}
double patch::get_Fy_B(int d, int i, int j, int k) const {
    return Fy[yfc_idx(i,j,k)].B[d];
}
double patch::get_Fz_B(int d, int i, int j, int k) const {
    return Fz[zfc_idx(i,j,k)].B[d];
}
// total flux accessors
const cons& patch::get_Fx(int i, int j, int k) const {
    return Fx[xfc_idx(i,j,k)];
}
const cons& patch::get_Fy(int i, int j, int k) const {
    return Fy[yfc_idx(i,j,k)];
}
const cons& patch::get_Fz(int i, int j, int k) const {
    return Fz[zfc_idx(i,j,k)];
}
cons& patch::get_Fx(int i, int j, int k) {
    return Fx[xfc_idx(i,j,k)];
}
cons& patch::get_Fy(int i, int j, int k) {
    return Fy[yfc_idx(i,j,k)];
}
cons& patch::get_Fz(int i, int j, int k) {
    return Fz[zfc_idx(i,j,k)];
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

// floor state protection
prim amrtree::pvfs(double r, double th) const {
    prim W;
    double rfscal = std::pow(r/r_floor_ref,-1.5);
    W.rho = rho_floor_r0*rfscal;
    W.eps = eps_floor_r0*rfscal;
    W.v[0] = W.v[1] = W.v[2] = 0.0;
    W.B[0] = W.B[1] = W.B[2] = 0.0;
    return W;
}

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
amrtree::amrtree(std::array<double,3> dom_l, std::array<double,3> dom_h, int nqlt, double M, double a, double Q, double gm) : mtr(M,a,Q), stt(gm), prmv(mtr,stt), cnsv(mtr,stt,prmv) {
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
// magnetic initialisation
void patch::B_init() {
    // initialise Bfx
    for(int i=-ghost; i<block+ghost+1; i++) {
        for(int j=-ghost; i<block+ghost; i++) {
            for(int k=-ghost; i<block+ghost; i++) {
                if(i>-ghost && i<block+ghost) {
                    Bfx[Bfx_idx(i,j,k)] = (cell_(i-1,j,k).W.B[0]+cell_(i,j,k).W.B[0])/2;
                } else if(i==-ghost) {
                    Bfx[Bfx_idx(i,j,k)] = cell_(i,j,k).W.B[0];
                } else {
                    Bfx[Bfx_idx(i,j,k)] = cell_(i-1,j,k).W.B[0];
                }
            }
        }
    }
    // initialise Bfy
    for(int i=-ghost; i<block+ghost; i++) {
        for(int j=-ghost; i<block+ghost+1; i++) {
            for(int k=-ghost; i<block+ghost; i++) {
                if(i>-ghost && i<block+ghost) {
                    Bfx[Bfx_idx(i,j,k)] = (cell_(i-1,j,k).W.B[0]+cell_(i,j,k).W.B[0])/2;
                } else if(i==-ghost) {
                    Bfx[Bfx_idx(i,j,k)] = cell_(i,j,k).W.B[0];
                } else {
                    Bfx[Bfx_idx(i,j,k)] = cell_(i-1,j,k).W.B[0];
                }
            }
        }
    }
    // initialise Bfz
    for(int i=-ghost; i<block+ghost; i++) {
        for(int j=-ghost; i<block+ghost; i++) {
            for(int k=-ghost; i<block+ghost+1; i++) {
                if(i>-ghost && i<block+ghost) {
                    Bfx[Bfx_idx(i,j,k)] = (cell_(i-1,j,k).W.B[0]+cell_(i,j,k).W.B[0])/2;
                } else if(i==-ghost) {
                    Bfx[Bfx_idx(i,j,k)] = cell_(i,j,k).W.B[0];
                } else {
                    Bfx[Bfx_idx(i,j,k)] = cell_(i-1,j,k).W.B[0];
                }
            }
        }
    }
}

// patch neighbour finding
patch* amrtree::nbhd(patch* p, int dim, int side) const {
    std::array<int,3> nb_ijk = p->ijk;
    nb_ijk[dim] += side;
    // search quilt for same-level patch
    for(auto& q : quilt) {
        if(q->lvl == p->lvl && q->ijk == nb_ijk) {
            return q.get();
        }
    }
    // if search fails, find coarser patch
    if(p->lvl>0) {
        std::array<int,3> crs_ijk;
        for(int i=0; i<3; i++) {
            crs_ijk[i] = p->ijk[i]/2;
        }
        crs_ijk[dim] += side;

        for(auto& q : quilt) {
            if(q->lvl == p->lvl-1 && q->ijk == crs_ijk) {
                return q.get();
            }
        }
    }
    // on the boundary return a null pointer (no cells past boundary)
    return nullptr;
}

// ghost processing
// copying cell data into ghosts
void amrtree::gh_copy(patch* p, patch* nb, int dim, int side) {
    // get start of indices in p and nb
    int pstart; int nbstart;
    if(side==1) {
        pstart = block;
        nbstart = 0;
    } else {
        pstart = -ghost;
        nbstart = block-ghost;
    }
    // iterate over ghost width and other two directions
    for(int g=0; g<ghost; g++) {
        for(int t1=-ghost; t1<block+ghost; t1++) {
            for(int t2=-ghost; t2<block+ghost; t2++) {
                int pi,pj,pk;
                int ni,nj,nk;
                // map coordinate indices by face direction
                if(dim==0) {
                    pi = pstart+g; pj = t1; pk = t2;
                    ni = nbstart+g; nj = t1; nk = t2;
                } else if(dim==1) {
                    pi = t1; pj = pstart+g; pk = t2;
                    ni = t1; nj = nbstart+g; nk = t2;
                } else {
                    pi = t1; pj = t2; pk = pstart+g;
                    ni = t1; nj = t2; nk = nbstart+g;
                }
                // boundary checks
                if(t1<0 || t1>=block || t2<0 || t2>=block) continue;
                // value copying
                p->cell_(pi,pj,pk).U = nb->cell_(ni,nj,nk).U;
                p->cell_(pi,pj,pk).W = nb->cell_(ni,nj,nk).W;
            }
        }
    }
}
// prolongation to interpolate ghosts into coarser neighbours
void amrtree::gh_prolong(patch* p, patch* nb_crs, int dim, int side) {
    int pstart = (side==1)? block : -ghost;
    for(int g=0; g<ghost; g++) {
        for(int t1=0; t1<block; t1++) {
            for(int t2=0; t2<block; t2++) {
                // determine cell to interpolate into
                int fi,fj,fk;
                if(dim==0) {
                    fi = pstart+g; fj = t1; fk = t2;
                } else if(dim==1) {
                    fi = t1; fj = pstart+g; fk = t2;
                } else {
                    fi = t1; fj = t2; fk = pstart+g;
                }
                // determine part of coarse cell interpolated into
                int ci = (dim==0)? g/2 : t1/2;
                int cj = (dim==1)? g/2 : (dim==0? t1/2 : t2/2);
                int ck = (dim==2)? g/2 : t2/2;
                // offsets
                double offg = (g%2==0)? -0.25 : 0.25;
                double offt1 = (t1%2==0)? -0.25 : 0.25;
                double offt2 = (t2%2==0)? -0.25 : 0.25;
                cell& cc = nb_crs->cell_(ci,cj,ck);
                // transverse slopes
                auto intp_sc = [&](double vm, double vc, double vp, double on, double ot1, double ot2, double vm1, double vp1, double vm2, double vp2) -> double {
                    double sn = minmod(vc-vm,vp-vc); // normal
                    double st1 = minmod(vc-vm1,vp1-vc);
                    double st2 = minmod(vc-vm2,vp2-vc);
                    return vc+sn*on+st1*ot1+st2*ot2;
                };
                // prolongate for each conserved variable
                auto get_c = [&](int di, int dj, int dk) -> cell& {
                    return nb_crs->cell_(ci+di,cj+dj,ck+dk);
                };
                // normal offsets
                double on, ot1, ot2;
                if(dim==0) {
                    on = offg; ot1 = offt1; ot2 = offt2;
                } else if(dim==1) {
                    on = offt1; ot1 = offg; ot2 = offt2;
                } else {
                    on = offt1; ot1 = offt2; ot2 = offg;
                }
                cell& p_cell = p->cell_(fi,fj,fk);
                // get conserved variables
                p_cell.U.D = intp_sc(get_c(-1,0,0).U.D, cc.U.D, get_c(1,0,0).U.D, on, ot1, ot2, get_c(0,-1,0).U.D, get_c(0,1,0).U.D, get_c(0,0,-1).U.D, get_c(0,0,1).U.D);
                p_cell.U.tau = intp_sc(get_c(-1,0,0).U.tau, cc.U.tau, get_c(1,0,0).U.tau, on, ot1, ot2, get_c(0,-1,0).U.tau, get_c(0,1,0).U.tau, get_c(0,0,-1).U.tau, get_c(0,0,1).U.tau);
                for(int i=0; i<3; i++) {
                    p_cell.U.B[i] = intp_sc(get_c(-1,0,0).U.B[i], cc.U.B[i], get_c(1,0,0).U.B[i], on, ot1, ot2, get_c(0,-1,0).U.B[i], get_c(0,1,0).U.B[i], get_c(0,0,-1).U.B[i], get_c(0,0,1).U.B[i]);
                    p_cell.U.S[i] = intp_sc(get_c(-1,0,0).U.S[i], cc.U.S[i], get_c(1,0,0).U.S[i], on, ot1, ot2, get_c(0,-1,0).U.S[i], get_c(0,1,0).U.S[i], get_c(0,0,-1).U.S[i], get_c(0,0,1).U.S[i]);
                }
                // primitive floors
                prim pv;
                bool ok = cnsv.ctop(p_cell.U,p_cell.r,p_cell.th,pv);
                if(ok) {
                    p_cell.W = pv;
                } else {
                    prim fl = pvfs(p_cell.r,p_cell.th);
                    fl.B[0] = p_cell.W.B[0]; fl.B[1] = p_cell.W.B[1]; fl.B[2] = p_cell.W.B[2];
                    p_cell.U = cnsv.ptoc(p_cell.W,p_cell.r,p_cell.th);
                }
            }
        }
    }
}
// ghost boundaries for when nbhd returns nullptr
void amrtree::gh_bndy(patch* p, int dim, int side) {
    int pstart = (side==1)? block : ghost;
    // iterate over cells
    for(int g=0; g<ghost; g++) {
        for(int t1=0; t1<block; t1++) {
            for(int t2=0; t2<block; t2++) {
                // ghost and active cell indices
                int gi, gj, gk;
                int ai, aj, ak;
                // casewise by dimension again
                if(dim==0) {
                    gi = pstart+g; gj = t1; gk = t2;
                    ai = (side==1)? block-1-g : g; aj = t1; ak = t2;
                } else if(dim==1) {
                    gi = t1; gj = pstart+g; gk = t2;
                    ai = t1; aj = (side==1)? block-1-g : g; ak = t2;
                } else {
                    gi = t1; gj = t2; gk = pstart+g;
                    ai = t1; aj = t2; ak = (side==2)? block-1-g : g;
                }
                // assignments
                cell& gc = p->cell_(gi,gj,gk);
                cell& ac = p->cell_(ai,aj,ak);
                // boundary conditions:
                gc.U = ac.U; gc.W = ac.W;
            }
        }
    }
}
// ghost processing for a given patch pointer p.
void amrtree::ghosts(patch* p) {
    for(int dim=0; dim<3; dim++) {
        for(int side : {-1,1}) {
            patch* nb = nbhd(p, dim, side);
            if(nb && nb->lvl == p->lvl) {
                gh_copy(p,nb,dim,side);
            } else if (nb && nb->lvl < p->lvl) {
                gh_prolong(p,nb,dim,side);
            } else if (!nb) {
                gh_bndy(p,dim,side);
            }
        }
    }
    // assign pointers and value
    for(int i=-ghost; i<block+ghost; i++) {
        for(int j=-ghost; j<block+ghost; j++) {
            for(int k=-ghost; k<block+ghost; k++) {
                cell& c = p->cell_(i,j,k);
                if(!c.ghost) continue;
                // primitives and floor protections
                prim pv;
                bool ok = cnsv.ctop(c.U,c.r,c.th,pv);
                if(ok) {
                    c.W = pv;
                } else {
                    prim fl = pvfs(c.r,c.th);
                    fl.B[0] = c.W.B[0]; fl.B[1] = c.W.B[1]; fl.B[2] = c.W.B[2];
                    c.U = cnsv.ptoc(c.W,c.r,c.th);
                }
            }
        }
    }
}