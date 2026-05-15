#pragma once
#include <cmath>
#include <vector>
#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include "cell.hpp"

// this document contains the structures for dealing with the grid,
// grid patches, and adaptive mesh refinement.

// sizing constants
static constexpr int block = 16; // active cells per patch per dimension
static constexpr int ghost = 2; // ghost halo width (= RK order)
static constexpr int cpn = block + 2*ghost;

// patch struct
struct patch {
    // hierarchy information
    int lvl;
    std::array<int,3> ijk;
    patch* parent;
    std::array<patch*,8> children;
    bool leaf;

    // edges
    std::vector<double> xedge; 
    std::vector<double> yedge; 
    std::vector<double> zedge;
    std::vector<cell> cells; // cells indexed within patch
    // fluxes in each direction
    std::vector<cons> Fx;
    std::vector<cons> Fy;
    std::vector<cons> Fz;

    // patch constructor
    patch(int lvl, std::array<int,3> ijk, std::array<double,3> cor1, std::array<double,3> cor2, patch* parent = nullptr);
    // cell indexing and constructors
    int lindx(int i, int j, int k) const;
    cell& cell_(int i, int j, int k); // non-constant cells
    const cell& cell_(int i, int j, int k) const; // constant cells

    // cell geometry
    std::array<double,3> cellmid(int i, int j, int k) const;
    std::array<double,3> corner(int i, int j, int k) const;
    double dx() const;
    double dy() const;
    double dz() const;
    void cell_init(); // geometry initialisation

    // direction of faces
    struct facedir {
        enum {X=0, Y=1, Z=2};
    };

    // face-centred magnetic field and edge-centred EMF
    std::vector<double> Bfx; std::vector<double> Bfy; std::vector<double> Bfz;
    std::vector<double> EMFx; std::vector<double> EMFy; std::vector<double> EMFz;
    int Bfx_idx(int i, int j, int k) const;
    int Bfy_idx(int i, int j, int k) const;
    int Bfz_idx(int i, int j, int k) const;
    int EMFx_idx(int i, int j, int k) const;
    int EMFy_idx(int i, int j, int k) const;
    int EMFz_idx(int i, int j, int k) const;
    void B_init(); // magnetic initialisation

    // face indexing
    int xfc_idx(int i, int j, int k) const;
    int yfc_idx(int i, int j, int k) const;
    int zfc_idx(int i, int j, int k) const;
    // flux accessors
    double get_Fx_B(int d, int i, int j, int k) const;
    double get_Fy_B(int d, int i, int j, int k) const;
    double get_Fz_B(int d, int i, int j, int k) const;
    const cons& get_Fx(int i, int j, int k) const;
    const cons& get_Fy(int i, int j, int k) const;
    const cons& get_Fz(int i, int j, int k) const;
    cons& get_Fx(int i, int j, int k);
    cons& get_Fy(int i, int j, int k);
    cons& get_Fz(int i, int j, int k);
};
// amr (adaptive mesh refinement) tree struct
class amrtree {
public:
    // physics
    metric mtr;
    state stt;
    primitive prmv;
    conserved cnsv;

    // constructor
    std::vector<std::unique_ptr<patch>> quilt; // quilt = collection of patches
    amrtree(std::array<double,3> dom_l, std::array<double,3> dom_h, int nqlt, double M, double a, double Q, double gm);
    patch* nbhd(patch* p, int dim, int side) const;
    
    // refinement functions
    void regrid();
    void step(double dt);
    void refine();
    void flatten();
private:
    // ghosts
    void ghosts(patch* p);
    void gh_copy(patch* p, patch* nb, int dim, int side);
    void gh_prolong(patch* p, patch* nb, int dim, int side);
    void gh_bndy(patch* p, int dim, int side);
    // floor states for when reconstruction of primitives fails
    prim pvfs(double r, double th) const;
    static constexpr double rho_floor_r0 = 1; // REDEFINE LATER
    static constexpr double eps_floor_r0 = 1; // REDEFINE LATER
    static constexpr double r_floor_ref = 1.0;
};