import numpy as np
from numba import njit, prange, float64
from numba_progress import ProgressBar
from pykdtree.kdtree import KDTree
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from render.Classes.base import *
import render.Classes.GravField.methods as GravMethod

@njit
def get_spacing(grid:np.ndarray):
    """Gives the average spacing of grid points in each coordinate direction.
    
    grid: np.ndarray with shape (N, 4) specifying the points of the grid."""

    scale = np.zeros(4, dtype=np.float64)
    for d in range(4):
        vals = np.sort(grid[:,d])
        diffs = np.diff(vals)
        diffs = diffs[diffs > 1e-12] # Remove duplicates
        if len(diffs) > 0: scale[d] = np.mean(diffs)
        else: scale[d] = 0.
    return scale

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

def generate_grid(center:Vec, bg_rad:float, n:float=40, max_levels:int=3, var_threshold:float=0.02):
    """Generate an adaptive grid that varies with the metric. For now, only applicable to static spacetimes.
    
    center, bg_rad: Center and radius of the background box.
    
    n: The initial number of points per coordinate direction.
    
    max_levels: Maximum number of recursions for point generation.
    
    var_threshold: Threshold of percentage variation of the metric to determine if more points are needed."""

    # Generate initial grid within background box
    bg_bb_min = center - bg_rad * Vec(1,1,1); bg_bb_max = center + bg_rad * Vec(1,1,1)
    T, X, Y, Z = np.meshgrid(0, np.linspace(bg_bb_min.x,bg_bb_max.x,n), np.linspace(bg_bb_min.y,bg_bb_max.y,n), np.linspace(bg_bb_min.z,bg_bb_max.z,n))
    grid = np.column_stack((T.ravel(), X.ravel(), Y.ravel(), Z.ravel()))
    grid = grid[np.linalg.norm(grid[:,1:4] - center.np_array(), axis=1) <= bg_rad]

    # Recursive generation of more points
    for level in range(max_levels):
        print(f"Generating level {level+1} of points...")
        N = grid.shape[0]; g = np.zeros((N, 4, 4))
        for i in range(N): g[i] = GravMethod.metric_at(grid[i])
        dist, idx = KDTree(grid).query(grid, k=8)

        with ProgressBar(total=len(grid)) as pbar:
            var = get_var(g, idx, pbar)

        high_var_pts = grid[var > var_threshold]
        if len(high_var_pts) == 0: break

        with ProgressBar(total=len(high_var_pts)) as pbar:
            new_pts = get_new_pts(high_var_pts, dist, pbar)

        grid = np.vstack((grid, new_pts))
        print(f"Grid now has {grid.shape[0]} points")

    chunk_size = 2500000
    unique_all = []
    for start in range(0, len(grid), chunk_size):
        chunk = grid[start:start+chunk_size]
        scaled_chunk = np.round(chunk / 1e-9).astype(np.int64)
        _, idx_chunk = np.unique(scaled_chunk, axis=0, return_index=True)
        unique_all.append(start + idx_chunk)
    unique_idx = np.concatenate(unique_all)
    grid = grid[unique_idx]
    return grid

@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def precomp_g(grid:np.ndarray, pbar:ProgressBar):
    """Precompute the metric tensor on a specified grid.
        
    grid: np.ndarray with shape (N, 4) specifying the points at which the metric is evaluated at, in Minkowski coordinates."""

    N = grid.shape[0]
    g = np.zeros((N, 10))
    pairs = np.array([(0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)])
    for i in prange(N):
        metric_pos = GravMethod.coord_pos(grid[i])
        g_eval = GravMethod.metric_at(metric_pos)
        g[i] = g_eval[pairs]
        pbar.update(1)
    return g

@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def precomp_Gamma(grid:np.ndarray, neighbors:np.ndarray, g:np.ndarray, pbar:ProgressBar):
    """Precompute the Christoffel symbols on a specified grid.
    
    grid: np.ndarray with shape (N, 4) specifying the points at which the metric is evaluated at, in Minkowski coordinates.
    
    neighbors: np.ndarray with shape (N, k) specifying the index (in grid) of the k-th closest neighbors to a given point.
    Make sure that no direction dominates the closest neighbors.
    
    g: np.ndarray with shape (N, 4, 4) specifying the metric tensor at each point on the grid."""

    N = grid.shape[0]
    g_pairs = np.array([(0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)])
    chr_sym_pairs = np.array([(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,1,1),(0,1,2),(0,1,3),(0,2,2),(0,2,3),(0,3,3),
                              (1,0,0),(1,0,1),(1,0,2),(1,0,3),(1,1,1),(1,1,2),(1,1,3),(1,2,2),(1,2,3),(1,3,3),
                              (2,0,0),(2,0,1),(2,0,2),(2,0,3),(2,1,1),(2,1,2),(2,1,3),(2,2,2),(2,2,3),(2,3,3),
                              (3,0,0),(3,0,1),(3,0,2),(3,0,3),(3,1,1),(3,1,2),(3,1,3),(3,2,2),(3,2,3),(3,3,3)])
    chr_sym = np.zeros((N, 4, 4, 4))
    
    for i in prange(N):

        # Computing derivative
        X = grid[neighbors[i]] - grid[i] # Shape (k, 4)
        dg = np.zeros((4,4,4), dtype=np.float64) # dg[k,i,j] = Derivative of g[i,j] with respect to x^k
        metric_pos = GravMethod.coord_pos(grid[i])
        for j in range(10):
            mu, nu = g_pairs[j]
            delta_g = g[neighbors[i], mu, nu] - g[i, mu, nu]
            grad_mink, _, _, _ = np.linalg.lstsq(X, delta_g)
            grad_coord = GravMethod.mink_vel(grad_mink, metric_pos) # Transform gradient to metric coordinates (covariant transformation, so mink_vel)
            dg[:,mu,nu] = grad_coord
            if mu != nu: dg[:,nu,mu] = grad_coord
        
        # Calculate Christoffel symbols
        g_eval = g[i]; g_inv_eval = np.linalg.inv(g_eval)
        for c in range(4):
            for a in range(4):
                for b in range(4):
                    s = 0
                    for d in range(4):
                        s += g_inv_eval[c,d] * (dg[a,b,d] + dg[b,a,d] - dg[d,a,b])
                    chr_sym[i,c,a,b] = s / 2

        pbar.update(1)
    return chr_sym

def precomp_all(grid:np.ndarray):
    print(f"Generating grid...")
    grid = generate_grid()
    os.makedirs(os.path.dirname("Points/grid_pts.npy") or ".", exist_ok=True)
    np.save("Points/grid_pts.npy", grid)

    print("Computing metric tensor...")
    with ProgressBar(total=grid.shape[0]) as pbar:
        g_grid = precomp_g(grid, pbar)
    os.makedirs(os.path.dirname("Points/metric_grid.npy") or ".", exist_ok=True)
    np.save("Points/metric_grid.npy", g_grid)

    # Compute isotropic grid (so that no direction dominates closest neighbors)
    print("Computing closest points...")
    scale = get_spacing(grid)
    grid_norm = grid / np.maximum(scale, 1e-12)
    _, neighbors = KDTree(grid_norm).query(grid_norm, k=20)

    print("Computing Christoffel symbols...")
    with ProgressBar(total=grid.shape[0]) as pbar:
        Gamma_grid = precomp_Gamma(grid, neighbors, g_grid, pbar)  
    os.makedirs(os.path.dirname("Points/chr_sym_grid.npy") or ".", exist_ok=True)
    np.save("Points/chr_sym_grid.npy", Gamma_grid)


if __name__ == "__main__":
    grid = generate_grid(center=Vec(10,0,0), bg_rad=20)
    print(grid.shape[0])