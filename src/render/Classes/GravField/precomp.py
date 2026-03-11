import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar

import Classes.GravField.methods as GravMethod

@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def precomp_g(grid:np.ndarray, pbar:ProgressBar):
    """Precompute the metric tensor on a specified grid.
        
    grid: np.ndarray with shape (N, 4) specifying the points at which the metric is evaluated at."""

    N = grid.shape[0]
    g = np.zeros((N, 4, 4))
    for i in prange(N):
        g_eval = GravMethod.metric_at(grid[i])
        g[i] = g_eval
        pbar.update(1)
    return g

@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def precomp_Gamma(grid:np.ndarray, neighbors:np.ndarray, g:np.ndarray, pbar:ProgressBar):
    """Precompute the Christoffel symbols on a specified grid.
    
    grid: np.ndarray with shape (N, 4) specifying the points at which the metric is evaluated at.
    
    neighbors: np.ndarray with shape (N, k) specifying the index (in grid) of the k-th closest neighbors to a given point. Obtained from:
    _, neighbors = scipy.spatial.ckdtree(grid).query(coords, k=k)
    
    g: np.ndarray with shape (N, 4, 4) specifying the metric tensor at each point on the grid."""


    N = grid.shape[0]
    pairs = np.array([(0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)])
    chr_sym = np.zeros((N, 4, 4, 4))
    
    for i in prange(N):

        # Computing derivative
        X = grid[neighbors[i]] - grid[i] # Shape (k, 4)
        dg = np.zeros((4,4,4), dtype=np.float64) # dg[k,i,j] = Derivative of g[i,j] with respect to x^k
        for j in range(10):
            mu, nu = pairs[j]
            delta_g = g[neighbors[i], mu, nu] - g[i, mu, nu]
            A, b = np.dot(X.T, X), np.dot(X.T, delta_g)
            A += 1e-10 * np.eye(4) # For preventing singular matrices
            grad = np.linalg.solve(A, b) # Gradient of g_mu,nu
            dg[:,mu,nu] = grad
            if mu != nu: dg[:,nu,mu] = grad
        
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

t, r, theta, phi = np.meshgrid(np.linspace(-10,10,7), np.linspace(0.01,25,80), np.linspace(0,np.pi,60), np.linspace(0,2*np.pi,90))
grid = np.column_stack(t.ravel(), r.ravel(), theta.ravel(), phi.ravel())
with ProgressBar(total=grid.shape[0]) as pbar:
    g_grid = precomp_g(grid, pbar)