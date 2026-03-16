import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
from pykdtree.kdtree import KDTree
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from render.Classes.base import *
from render.Classes.GravField.funcs import *

@njit
def metric_at(x:np.ndarray):
    """Returns the metric tensor evaluated at x. Assumes g has an analytic expression written to file.
    
    x: A point in spacetime described in metric coordinates."""

    g = np.array([[g00(x[0], x[1], x[2], x[3]), g01(x[0], x[1], x[2], x[3]), g02(x[0], x[1], x[2], x[3]), g03(x[0], x[1], x[2], x[3])],
                  [g01(x[0], x[1], x[2], x[3]), g11(x[0], x[1], x[2], x[3]), g12(x[0], x[1], x[2], x[3]), g13(x[0], x[1], x[2], x[3])],
                  [g02(x[0], x[1], x[2], x[3]), g12(x[0], x[1], x[2], x[3]), g22(x[0], x[1], x[2], x[3]), g23(x[0], x[1], x[2], x[3])],
                  [g03(x[0], x[1], x[2], x[3]), g13(x[0], x[1], x[2], x[3]), g23(x[0], x[1], x[2], x[3]), g33(x[0], x[1], x[2], x[3])]], dtype=float64)
    return g

@njit
def coord_deriv(grad:np.ndarray, pos:np.ndarray) -> np.ndarray:
    """vec: A contravariant four-vector described in metric coordinates (relative to a stationary observer at infinity).

    pos: A spacetime point described in metric coordinates (relative to a stationary observer at infinity).

    Performs a contravariant transformation of vec at pos from metric coordinates to Minkowski coordinates."""

    j0 = [Jinv00(pos[0], pos[1], pos[2], pos[3]), Jinv01(pos[0], pos[1], pos[2], pos[3]), Jinv02(pos[0], pos[1], pos[2], pos[3]), Jinv03(pos[0], pos[1], pos[2], pos[3])]
    j1 = [Jinv10(pos[0], pos[1], pos[2], pos[3]), Jinv11(pos[0], pos[1], pos[2], pos[3]), Jinv12(pos[0], pos[1], pos[2], pos[3]), Jinv13(pos[0], pos[1], pos[2], pos[3])]
    j2 = [Jinv20(pos[0], pos[1], pos[2], pos[3]), Jinv21(pos[0], pos[1], pos[2], pos[3]), Jinv22(pos[0], pos[1], pos[2], pos[3]), Jinv23(pos[0], pos[1], pos[2], pos[3])]
    j3 = [Jinv30(pos[0], pos[1], pos[2], pos[3]), Jinv31(pos[0], pos[1], pos[2], pos[3]), Jinv32(pos[0], pos[1], pos[2], pos[3]), Jinv33(pos[0], pos[1], pos[2], pos[3])]

    new_grad = [j0[0]*grad[0] + j0[1]*grad[1] + j0[2]*grad[2] + j0[3]*grad[3],
                j1[0]*grad[0] + j1[1]*grad[1] + j1[2]*grad[2] + j1[3]*grad[3],
                j2[0]*grad[0] + j2[1]*grad[1] + j2[2]*grad[2] + j2[3]*grad[3],
                j3[0]*grad[0] + j3[1]*grad[1] + j3[2]*grad[2] + j3[3]*grad[3]]
    return np.array(new_grad, dtype=float64)

@njit
def coord_pos(pos:np.ndarray) -> np.ndarray:
    """pos: A spacetime event described in Minkowski coordinates (relative to a stationary observer at infinity).

    Performs a coordinate transformation from Minkowski coordinates to that of the gravitational field's metric."""

    return np.array([X0(pos[0], pos[1], pos[2], pos[3]),
                     X1(pos[0], pos[1], pos[2], pos[3]),
                     X2(pos[0], pos[1], pos[2], pos[3]),
                     X3(pos[0], pos[1], pos[2], pos[3])], dtype=float64)

@njit(parallel=True, fastmath=True)
def get_var(g_grid:np.ndarray, neigh_idx:np.ndarray, pbar:ProgressBar):
    N = g_grid.shape[0]
    var = np.zeros(N)
    for i in prange(N):
        neigh_g = g_grid[neigh_idx[i, 1:]]
        mean_g = g_grid[i].mean()
        if np.abs(mean_g) > 1e-8: # Use percentage variation
            var[i] = np.max(np.abs(neigh_g - np.broadcast_to(g_grid[i], neigh_g.shape))) / np.abs(mean_g) 
        else:
            var[i] = np.max(np.abs(neigh_g - np.broadcast_to(g_grid[i], neigh_g.shape)))
        pbar.update(1)
    return var

@njit(parallel=True, fastmath=True)
def get_new_pts(high_var_pts:np.ndarray, dist:np.ndarray, pbar:ProgressBar):
    N = high_var_pts.shape[0]
    new_pts = np.empty((N*8, 4), dtype=np.float64)

    for i in prange(N):
        dx = dist[i,1] / 4 # Estimation of local cell size
        vals = np.array([-dx, dx]); idx = 8*i
        
        k = 0
        for j in range(2):
            for m in range(2):
                for n in range(2):
                    new_pts[idx+k, 0] = high_var_pts[i][0]
                    new_pts[idx+k, 1] = high_var_pts[i][1] + vals[j]
                    new_pts[idx+k, 2] = high_var_pts[i][2] + vals[m]
                    new_pts[idx+k, 3] = high_var_pts[i][3] + vals[n]
                    k += 1
        pbar.update(1)
    return new_pts

def generate_grid(center:Vec, bg_rad:float, n:float=10, max_levels:int=3,
                  var_threshold:float=0.02, tol:float=1e-9, chunk_size:int=2500000):
    """Generate an adaptive grid that varies with the metric. For now, only applicable to static spacetimes.
    
    center, bg_rad: Center and radius of the background box.
    
    n: The initial number of points per coordinate direction.
    
    max_levels: Maximum number of recursions for point generation.
    
    var_threshold: Threshold of percentage variation of the metric to determine if more points are needed.
    
    tol: Points within this distance of each other will be considered duplicates and be removed."""

    # Generate initial grid within background box
    bg_bb_min = center - bg_rad * Vec(1,1,1); bg_bb_max = center + bg_rad * Vec(1,1,1)
    T, X, Y, Z = np.meshgrid(0, np.linspace(bg_bb_min.x,bg_bb_max.x,n), np.linspace(bg_bb_min.y,bg_bb_max.y,n), np.linspace(bg_bb_min.z,bg_bb_max.z,n))
    grid = np.column_stack((T.ravel(), X.ravel(), Y.ravel(), Z.ravel()))
    grid = grid[np.linalg.norm(grid[:,1:4] - center.np_array(), axis=1) <= bg_rad]

    # Recursive generation of more points
    for level in range(max_levels):
        print(f"Generating level {level+1} of points...")
        N = grid.shape[0]; g = np.zeros((N, 4, 4))
        for i in range(N): g[i] = metric_at(grid[i])
        dist, idx = KDTree(grid).query(grid, k=8)

        with ProgressBar(total=len(grid)) as pbar:
            var = get_var(g, idx, pbar)

        high_var_pts = grid[var > var_threshold]
        if len(high_var_pts) == 0: break

        with ProgressBar(total=len(high_var_pts)) as pbar:
            new_pts = get_new_pts(high_var_pts, dist, pbar)

        grid = np.vstack((grid, new_pts))

    unique_all = []
    for start in range(0, len(grid), chunk_size):
        end = min(start+chunk_size, len(grid)-1)
        chunk = grid[start:end]
        scaled_chunk = np.round(chunk / tol).astype(np.int64)
        _, idx_chunk = np.unique(scaled_chunk, axis=0, return_index=True)
        unique_all.append(start + idx_chunk)
    unique_idx = np.concatenate(unique_all)
    grid = grid[unique_idx]
    print(f"Grid has {len(grid)} points")
    return grid

@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def precomp_g(grid:np.ndarray, pbar:ProgressBar):
    """Precompute the metric tensor on a specified grid.
        
    grid: np.ndarray with shape (N, 4) specifying the points at which the metric is evaluated at, in Minkowski coordinates."""

    N = grid.shape[0]
    g = np.zeros((N, 10))
    pairs = np.array([(0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)])
    for i in prange(N):
        metric_pos = coord_pos(grid[i])
        g_eval = metric_at(metric_pos)
        for j in range(10):
            mu, nu = pairs[j]
            g[i,j] = g_eval[mu,nu]
        pbar.update(1)
    return g

@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def precomp_Gamma(grid:np.ndarray, neighbors:np.ndarray, g:np.ndarray, pbar:ProgressBar):
    """Precompute the Christoffel symbols on a specified grid.
    
    grid: np.ndarray with shape (N, 4) specifying the points at which the metric is evaluated at, in Minkowski coordinates.
    
    neighbors: np.ndarray with shape (N, k) specifying the index (in grid) of the k-th closest neighbors to a given point.
    Make sure that no direction dominates the closest neighbors.
    
    g: np.ndarray with shape (N, 10) specifying the metric tensor at each point on the grid."""

    N = grid.shape[0]
    g_pairs = np.array([(0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)])
    chr_sym_pairs = np.array([(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,1,1),(0,1,2),(0,1,3),(0,2,2),(0,2,3),(0,3,3),
                              (1,0,0),(1,0,1),(1,0,2),(1,0,3),(1,1,1),(1,1,2),(1,1,3),(1,2,2),(1,2,3),(1,3,3),
                              (2,0,0),(2,0,1),(2,0,2),(2,0,3),(2,1,1),(2,1,2),(2,1,3),(2,2,2),(2,2,3),(2,3,3),
                              (3,0,0),(3,0,1),(3,0,2),(3,0,3),(3,1,1),(3,1,2),(3,1,3),(3,2,2),(3,2,3),(3,3,3)])
    chr_sym = np.zeros((N,4,4,4)); chr_sym_reduct = np.zeros((N,40))
    
    for i in prange(N):

        # Computing derivative
        X = grid[neighbors[i]] - grid[i] # Shape (k, 4)
        dg = np.zeros((4,4,4), dtype=np.float64) # dg[k,i,j] = Derivative of g[i,j] with respect to x^k
        metric_pos = coord_pos(grid[i])
        for j in range(10):
            delta_g = g[neighbors[i], j] - g[i, j]
            grad_mink, _, _, _ = np.linalg.lstsq(X, delta_g)
            grad_coord = coord_deriv(grad_mink, metric_pos) # Transform gradient to metric coordinates (covariant transformation, so mink_vel)
            mu, nu = g_pairs[j]
            dg[:,mu,nu] = grad_coord
            if mu != nu: dg[:,nu,mu] = grad_coord
        
        # Calculate Christoffel symbols
        g00, g01, g02, g03, g11, g12, g13, g22, g23, g33 = g[i]
        g_eval_m = np.array([[g00, g01, g02, g03],
                             [g01, g11, g12, g13],
                             [g02, g12, g22, g23],
                             [g03, g13, g23, g33]], dtype=np.float64)
        g_inv_eval = np.linalg.inv(g_eval_m)
        for c in range(4):
            for a in range(4):
                for b in range(4):
                    s = 0
                    for d in range(4):
                        s += g_inv_eval[c,d] * (dg[a,b,d] + dg[b,a,d] - dg[d,a,b])
                    chr_sym[i,c,a,b] = s / 2

        for j in range(40):
            c, a, b = chr_sym_pairs[j]
            chr_sym_reduct[i,j] = chr_sym[i,c,a,b]
        pbar.update(1)
    return chr_sym_reduct

def precomp_all(save:bool=False, k:int=20, **kwargs):
    print(f"Generating grid...")
    grid = generate_grid(**kwargs)

    print("Computing metric tensor...")
    with ProgressBar(total=grid.shape[0]) as pbar:
        g_grid = precomp_g(grid, pbar)

    print("Computing Christoffel symbols...")
    _, neighbors = KDTree(grid).query(grid, k=k)
    with ProgressBar(total=grid.shape[0]) as pbar:
        Gamma_grid = precomp_Gamma(grid, neighbors, g_grid, pbar)  

    if save:
        os.makedirs(os.path.dirname(f"Points/grid_pts.npy") or ".", exist_ok=True)
        os.makedirs(os.path.dirname(f"Points/neigh_pts.npy") or ".", exist_ok=True)
        os.makedirs(os.path.dirname(f"Points/metric_grid.npy") or ".", exist_ok=True)
        os.makedirs(os.path.dirname(f"Points/chr_sym_grid.npy") or ".", exist_ok=True)
        np.save(f"Points/grid_pts.npy", grid)
        np.save(f"Points/neigh_pts.npy", neighbors)
        np.save(f"Points/metric_grid.npy", g_grid)
        np.save(f"Points/chr_sym_grid.npy", Gamma_grid)
