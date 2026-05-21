#include <cmath>
#include <vector>
#include <array>
#include <memory>
#include "grid.hpp"
#include "cell.hpp"
#include "metric.hpp"
#include "initial.hpp"
#include "hlld.hpp"

// this document contains the methods for dealing with the grid and
// adaptive mesh refinement (amr). the grid is split up into a quilt
// of patches, with each patch containing cells which hold data.

using namespace grid;

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
    cs_x.resize((cpn+1)*cpn*cpn);
    cs_y.resize(cpn*(cpn+1)*cpn);
    cs_z.resize(cpn*cpn*(cpn+1));
    
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
    W.T = T_floor;
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
amrtree::amrtree(std::array<double,3> dom_l, std::array<double,3> dom_h, int nqlt, double M, double a, double Q, double gm) : mtr(M,a,Q), stt(gm), prmv(mtr,stt), cnsv(mtr,stt) {
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
    // set global maximum density for initialisation
    double glmx_rho = 0.0;
    for(auto& p : quilt) {
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    glmx_rho = std::max(glmx_rho, p->cell_(i,j,k).W.rho);
                }
            }
        }
    }
}
// magnetic initialisation
void patch::B_init() {
    // initialise Bfx
    for(int i=-ghost; i<block+ghost+1; i++) {
        for(int j=-ghost; j<block+ghost; j++) {
            for(int k=-ghost; k<block+ghost; k++) {
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
        for(int j=-ghost; j<block+ghost+1; j++) {
            for(int k=-ghost; k<block+ghost; k++) {
                if(j>-ghost && j<block+ghost) {
                    Bfy[Bfy_idx(i,j,k)] = (cell_(i,j-1,k).W.B[1]+cell_(i,j,k).W.B[1])/2;
                } else if(j==-ghost) {
                    Bfy[Bfy_idx(i,j,k)] = cell_(i,j,k).W.B[1];
                } else {
                    Bfy[Bfy_idx(i,j,k)] = cell_(i,j-1,k).W.B[1];
                }
            }
        }
    }
    // initialise Bfz
    for(int i=-ghost; i<block+ghost; i++) {
        for(int j=-ghost; j<block+ghost; j++) {
            for(int k=-ghost; k<block+ghost+1; k++) {
                if(k>-ghost && k<block+ghost) {
                    Bfz[Bfz_idx(i,j,k)] = (cell_(i,j,k-1).W.B[2]+cell_(i,j,k).W.B[2])/2;
                } else if(i==-ghost) {
                    Bfz[Bfz_idx(i,j,k)] = cell_(i,j,k).W.B[2];
                } else {
                    Bfz[Bfz_idx(i,j,k)] = cell_(i,j,k-1).W.B[2];
                }
            }
        }
    }
}

// flux computation
void patch::fluxcomp(const metric& mtr, const state& stt) {
    conserved cv(mtr,stt);
    // x-direction fluxes
    for(int i=-1; i<block; i++) {
        for(int j=0; j<block; j++) {
            for(int k=0; k<block; k++) {
                facestate fs = reconfp(*this,i,j,k,0);
                // overwrite magnetic fluxes for transport constraint
                fs.L.B[0] = Bfx[Bfx_idx(i+1,j,k)];
                fs.R.B[0] = Bfx[Bfx_idx(i+1,j,k)];
                // face metric
                const cell& cL = cell_(i,j,k);
                const cell& cR = cell_(i+1,j,k);
                double rf = (cL.r+cR.r)/2;
                double thf = (cL.th+cR.th)/2;
                // conserved fluxes
                cons UL = cv.ptoc(fs.L,rf,thf);
                cons UR = cv.ptoc(fs.R,rf,thf);
                // hlld call
                hlldflux flux = hlld(fs.L,fs.R,UL,UR,mtr,rf,thf,stt,0);
                // flux storage
                get_Fx(i,j,k) = flux.F;
                cs_x[xfc_idx(i,j,k)] = flux.ctspeed;
            }
        }
    }
    // y-direction fluxes
    for(int i=0; i<block; i++) {
        for(int j=-1; j<block; j++) {
            for(int k=0; k<block; k++) {
                facestate fs = reconfp(*this,i,j,k,1);
                // overwrite magnetic fluxes for transport constraint
                fs.L.B[1] = Bfy[Bfy_idx(i,j+1,k)];
                fs.R.B[1] = Bfy[Bfy_idx(i,j+1,k)];
                // face metric
                const cell& cL = cell_(i,j,k);
                const cell& cR = cell_(i,j+1,k);
                double rf = (cL.r+cR.r)/2;
                double thf = (cL.th+cR.th)/2;
                // conserved fluxes
                cons UL = cv.ptoc(fs.L,rf,thf);
                cons UR = cv.ptoc(fs.R,rf,thf);
                // hlld call
                hlldflux flux = hlld(fs.L,fs.R,UL,UR,mtr,rf,thf,stt,0);
                // flux storage
                get_Fy(i,j,k) = flux.F;
                cs_y[yfc_idx(i,j,k)] = flux.ctspeed;
            }
        }
    }
    // z-direction fluxes
    for(int i=0; i<block; i++) {
        for(int j=0; j<block; j++) {
            for(int k=-1; k<block; k++) {
                facestate fs = reconfp(*this,i,j,k,2);
                // overwrite magnetic fluxes for transport constraint
                fs.L.B[2] = Bfz[Bfz_idx(i,j,k+1)];
                fs.R.B[2] = Bfz[Bfz_idx(i,j,k+1)];
                // face metric
                const cell& cL = cell_(i,j,k);
                const cell& cR = cell_(i,j,k+1);
                double rf = (cL.r+cR.r)/2;
                double thf = (cL.th+cR.th)/2;
                // conserved fluxes
                cons UL = cv.ptoc(fs.L,rf,thf);
                cons UR = cv.ptoc(fs.R,rf,thf);
                // hlld call
                hlldflux flux = hlld(fs.L,fs.R,UL,UR,mtr,rf,thf,stt,0);
                // flux storage
                get_Fz(i,j,k) = flux.F;
                cs_z[zfc_idx(i,j,k)] = flux.ctspeed;
            }
        }
    }
}
// enforcement of floor values
void patch::floors(amrtree& tree, patch& p, const state& stt, const metric& mtr) {
    for(int i=0; i<block; i++) {
        for(int j=0; j<block; j++) {
            for(int k=0; k<block; k++) {
                cell& c = p.cell_(i,j,k);
                // check for unphysical conserved variables
                bool bhl = !std::isfinite(c.U.D) || !std::isfinite(c.U.tau) || c.U.D<=0.0 || c.U.tau<=0.0;
                if(!bhl) {
                    // if physical, attempt reconstruction
                    prim pv;
                    bhl = !tree.cnsv.ctop(c.U,c.r,c.th,pv);
                    if(!bhl) {
                        // check physicality of primitives
                        bhl = (pv.rho<=0.0 || pv.eps<=0.0);
                        if (!bhl) c.W = pv;
                    }
                }
                // if unphysical, apply floors
                if(bhl || c.W.rho<tree.rho_floor_r0) {
                    prim fl = tree.pvfs(c.r,c.th);
                    fl.B[0] = c.W.B[0]; fl.B[1] = c.W.B[1]; fl.B[2] = c.W.B[2];
                    c.W = fl;
                    c.U = tree.cnsv.ptoc(c.W,c.r,c.th);
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
    int pstart = (side==1)? block : -ghost;
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
                    ai = t1; aj = t2; ak = (side==1)? block-1-g : g;
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

// refinement
void amrtree::refine(patch* p) {
    // exclude non-leaf patches
    if(!p->leaf) return;
    // divorce parent into eight octant children at edge midpoints
    double xm = (p->xedge[0]+p->xedge[block])/2;
    double ym = (p->yedge[0]+p->yedge[block])/2;
    double zm = (p->zedge[0]+p->zedge[block])/2;
    double xl = p->xedge[0]; double xh = p->xedge[block];
    double yl = p->yedge[0]; double yh = p->yedge[block];
    double zl = p->zedge[0]; double zh = p->zedge[block];
    // birth children
    std::array<double,3> cc1[8], cc2[8];
    cc1[0] = {xl,yl,zl}; cc2[0] = {xm,ym,zm};
    cc1[1] = {xl,yl,zm}; cc2[1] = {xm,ym,zh};
    cc1[2] = {xl,ym,zl}; cc2[2] = {xm,yh,zm};
    cc1[3] = {xl,ym,zm}; cc2[3] = {xm,yh,zh};
    cc1[4] = {xm,yl,zl}; cc2[4] = {xh,ym,zm};
    cc1[5] = {xm,yl,zm}; cc2[5] = {xh,ym,zh};
    cc1[6] = {xm,ym,zl}; cc2[6] = {xh,yh,zm};
    cc1[7] = {xm,ym,zm}; cc2[7] = {xh,yh,zh};
    // indexing children offset by octant
    for(int i=0; i<8; i++) {
        int ix = (i>>2)&1, iy = (i>>1)&1, iz = i&1;
        std::array<int,3> cijk = {2*p->ijk[0]+ix,2*p->ijk[1]+iy,2*p->ijk[2]+iz};
        quilt.push_back(std::make_unique<patch>(p->lvl+1,cijk,cc1[i],cc2[i],p));
        p->children[i] = quilt.back().get();
    }

    // prolongate data from parent into children
    for(int h=0; h<8; h++) {
        patch* child = p->children[h];
        int ix = (h>>2)&1, iy = (h>>1)&1, iz = h&1;
        // map child index to parent index
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    cell& cc = child->cell_(i,j,k);
                    int pi = (i+ix*block)/2, pj = (j+iy*block)/2, pk = (k+iz*block)/2;
                    // offset within coarse cell
                    double ox = (i%2==0)? -0.25 : 0.25;
                    double oy = (j%2==0)? -0.25 : 0.25;
                    double oz = (k%2==0)? -0.25 : 0.25;
                    // cell linear indexing
                    cell& pc = p->cell_(pi,pj,pk);
                    cell& ppx = p->cell_(pi+1,pj,pk);
                    cell& pmx = p->cell_(pi-1,pj,pk);
                    cell& ppy = p->cell_(pi,pj+1,pk);
                    cell& pmy = p->cell_(pi,pj-1,pk);
                    cell& ppz = p->cell_(pi,pj,pk+1);
                    cell& pmz = p->cell_(pi,pj,pk-1);

                    // prolong variables under slope limits
                    auto prolong = [&](double vc, double vmx, double vpx, double vmy, double vpy, double vmz, double vpz) {
                        double sx = minmod(vc-vmx,vpx-vc);
                        double sy = minmod(vc-vmy,vpy-vc);
                        double sz = minmod(vc-vmz,vpz-vc);
                        return vc+sx*ox+sy*oy+sz*oz; 
                    };
                    // interpolate conserved variables
                    cc.U.D = prolong(pc.U.D,ppx.U.D,pmx.U.D,ppy.U.D,pmy.U.D,ppz.U.D,pmz.U.D);
                    cc.U.tau = prolong(pc.U.tau,ppx.U.tau,pmx.U.tau,ppy.U.tau,pmy.U.tau,ppz.U.tau,pmz.U.tau);
                    for(int l=0; l<3; l++) {
                        cc.U.S[l] = prolong(pc.U.S[l],ppx.U.S[l],pmx.U.S[l],ppy.U.S[l],pmy.U.S[l],ppz.U.S[l],pmz.U.S[l]);
                        cc.U.B[l] = prolong(pc.U.B[l],ppx.U.B[l],pmx.U.B[l],ppy.U.B[l],pmy.U.B[l],ppz.U.B[l],pmz.U.B[l]);
                    }
                    // recover primitive variables
                    prim pv;
                    bool ok = cnsv.ctop(cc.U,cc.r,cc.th,pv);
                    if(ok) cc.W = pv;
                    else {
                        prim fl = pvfs(cc.r,cc.th);
                        fl.B[0] = cc.W.B[0]; fl.B[1] = cc.W.B[1]; fl.B[2] = cc.W.B[2];
                        cc.W = fl;
                        cc.U = cnsv.ptoc(cc.W,cc.r,cc.th);
                    }
                }
            }
        }
        child->B_init();
    }
    p->leaf = false;
}
// reconstruction of coarse cell by conservative restriction
void amrtree::ccrstr(patch* p) {
    // average values of conserved variables over children for parent
    for(int pi=0; pi<block; pi++) {
        for(int pj=0; pj<block; pj++) {
            for(int pk=0; pk<block; pk++) {
                // initialise parent
                cell& pc = p->cell_(pi,pj,pk);
                pc.U = cons();
                double volt = 0.0;
                // iterate over children
                for(int h=0; h<8; h++) {
                    patch* child = p->children[h];
                    if(!child) continue;
                    int ix = (h>>2)&1, iy = (h>>1)&1, iz = h&1;
                    for(int di=0; di<2; di++) {
                        for(int dj=0; dj<2; dj++) {
                            for(int dk=0; dk<2; dk++) {
                                // map parent to children
                                int fi = 2*(pi-ix*(block/2))+di;
                                int fj = 2*(pj-iy*(block/2))+dj;
                                int fk = 2*(pk-iz*(block/2))+dk;
                                // limit to fine cells in octant only
                                if(fi<0 || fi>=block || fj<0 || fj>=block || fk<0 || fk>=block) continue;
                                // volume and conserved variables
                                cell& fc = child->cell_(fi,fj,fk);
                                double vol = fc.vol;
                                volt += vol;
                                pc.U.D += vol*fc.U.D;
                                pc.U.tau += vol*fc.U.tau;
                                for(int a=0; a<3; a++) {
                                    pc.U.S[a] += vol*fc.U.S[a];
                                    pc.U.B[a] += vol*fc.U.B[a];
                                }
                            }
                        }
                    }
                }
                // volume normalisation
                if(volt>0.0) {
                    double inv = 1.0/volt;
                    pc.U.D *= inv; pc.U.tau *= inv;
                    for(int b=0; b<3; b++) {
                        pc.U.S[b] *= inv;
                        pc.U.B[b] *= inv;
                    }
                }
                // recover primitives
                prim pv;
                bool ok = cnsv.ctop(pc.U,pc.r,pc.th,pv);
                if(ok) pc.W = pv;
                else {
                    prim fl = pvfs(pc.r,pc.th);
                    fl.B[0] = pc.W.B[0]; fl.B[1] = pc.W.B[1]; fl.B[2] = pc.W.B[2];
                    pc.W = fl;
                    pc.U = cnsv.ptoc(pc.W,pc.r,pc.th);
                }
            }
        }
    }
    // initialise magnetic field
    p->B_init();
}
// flattening
void amrtree::flatten(patch* p) {
    if(p->leaf) return;
    // ensure children are leaves and construct data
    for(int h=0; h<8; h++) {
        if(p->children[h] && !p->children[h]->leaf) return;
    }
    ccrstr(p);
    // kill children to save memory
    for(int h=0; h<8; h++) {
        if(!p->children[h]) continue;
        // remove from quilt
        quilt.erase(std::remove_if(quilt.begin(),quilt.end(),[&](const std::unique_ptr<patch>& q){return q.get()==p->children[h];}),quilt.end());
        p->children[h] = nullptr;
    }
    // new leaf
    p->leaf = true;
}

// regrid based on refinement and flattening thresholds
void amrtree::regrid() {
    // thresholds and things
    static constexpr double lim_refine = 0.5; // relative gradient lower threshold for refinement
    static constexpr double lim_flatten = 0.1; // relative gradient upper threshold for flattening
    static constexpr int max_amr_lvl = 4; // maximum AMR refinement level
    static constexpr double horizon_refine_r0 = 3.0; // always refine at points less than ISCO
    // flag cells for refinement
    for(auto& p : quilt) {
        if(!p->leaf) continue;
        // useful state variables
        bool anyref = false;
        bool allflt = true;
        // iterate
        for(int i=0; i<block; i++) {
            for(int j=0; j<block; j++) {
                for(int k=0; k<block; k++) {
                    cell& c = p->cell_(i,j,k);
                    c.refine = false;
                    c.erase = false;
                    // mandate refinement near horizon
                    if(c.r<horizon_refine_r0 && p->lvl<max_amr_lvl) {
                        c.refine = true;
                        anyref = true;
                        allflt = false;
                        continue;
                    }
                    // density gradient
                    double dx = p->dx(), dy = p->dy(), dz = p->dz();
                    double chro = c.W.rho;
                    if(chro>0.0) {
                        // central difference for gradient
                        double drho_dx = std::abs(p->cell_(i+1,j,k).W.rho-p->cell_(i-1,j,k).W.rho)/(2*dx);
                        double drho_dy = std::abs(p->cell_(i,j+1,k).W.rho-p->cell_(i,j-1,k).W.rho)/(2*dy);
                        double drho_dz = std::abs(p->cell_(i,j,k+1).W.rho-p->cell_(i,j,k-1).W.rho)/(2*dz);
                        double drho = std::sqrt(drho_dx*drho_dx+drho_dy*drho_dy+drho_dz*drho_dz);
                        double rgrh = drho*dx/chro;
                        // threshold checks
                        if(rgrh>lim_refine && p->lvl<max_amr_lvl) {
                            c.refine = true;
                            anyref = true;
                            allflt = false;
                        }
                        if(rgrh>lim_flatten) {
                            allflt = false;
                        }
                    }
                    // magnetic gradient
                    double Bmag = std::sqrt(c.W.B[0]*c.W.B[0]+c.W.B[1]*c.W.B[1]+c.W.B[2]*c.W.B[2]);
                    if(Bmag>1e-14) {
                        double dB_dx = std::abs(p->cell_(i+1,j,k).W.B[0]-p->cell_(i-1,j,k).W.B[0])/(2*dx);
                        double dB_dy = std::abs(p->cell_(i,j+1,k).W.B[1]-p->cell_(i,j-1,k).W.B[1])/(2*dy);
                        double dB_dz = std::abs(p->cell_(i,j,k+1).W.B[2]-p->cell_(i,j,k-1).W.B[2])/(2*dz);
                        double rdB = (dB_dx*dx+dB_dy*dy+dB_dz*dz)/Bmag;
                        if(rdB>lim_refine && p->lvl<max_amr_lvl) {
                            c.refine = true;
                            anyref = true;
                            allflt = false;
                        }
                    }
                    // flag for possible flattening
                    if(!c.refine) c.erase = allflt;
                }
            }
        }
        if(anyref && p->lvl<max_amr_lvl) {
            p->refine = true;
        }
        if(allflt && p->lvl>0) {
            p->erase = true;
        }
    }
    // pass again and refine patches, collecting patches to refine
    std::vector<patch*> rfnQ, fltQ;
    for(auto& p : quilt) {
        if(p->leaf && p->refine) rfnQ.push_back(p.get());
        if(p->leaf && p->erase) fltQ.push_back(p.get());
    }
    // execute refinement and flattening
    for(patch* p : rfnQ) {
        refine(p);
        p->refine = false;
    }
    // flatten iff parent exists and no sibling needs to be refined
    for(patch* p : fltQ) {
        if(p->parent && p->parent->leaf==false) {
            bool dhl = true;
            for(int h=0; h<8; h++) {
                patch* sib = p->parent->children[h];
                if(sib && (!sib->leaf || sib->refine)) {
                    dhl = false; break;
                }
            }
            if(dhl) flatten(p->parent);
        }
        p->erase = false;
    }
}
// runge-kutta timestep to advance the grid
void amrtree::step(double dt) {
    dt = rk.step(*this); // call integrator
    regrid(); // regrid after each step
}