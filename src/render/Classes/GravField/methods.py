import numpy as np
from numba import njit

from render.Classes.base import *
from render.Classes.GravField.funcs import *
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Numba-jitted functions
@njit(fastmath=True)
def metric_interp(x:np.ndarray, grid:np.ndarray, g_grid:np.ndarray, neighbors:np.ndarray,
                  k:int=12, power:float=2.0, eps:float=1e-12):
    """Returns the interpolated metric at x. Call only after grid points are saved to file.
    
    x: A point in spacetime described in metric coordinates.
    
    g_grid: The metric tensor (N, 10) evaluated on a selection of grid points.
    
    neighbors: np.ndarray with shape (N, k) specifying the index (in grid) of the k-th closest neighbors to a given point.
    Make sure that no direction dominates the closest neighbors.
    
    power: Describes how quickly the weights of neighboring points fall off with distance."""

    assert k <= neighbors.shape[1]
    pairs = np.array([(0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)])
    dists = np.empty(k, dtype=np.float64); weights = np.empty(k, dtype=np.float64)

    for i in range(k):
        idx = neighbors[i]
        dp = grid[idx] - x
        d = np.sqrt(np.dot(dp, dp) + eps)
        dists[i] = d
        weights[i] = 1/(d**power) if d > 1e-20 else 1e20

    sum_w = np.sum(weights)
    if sum_w < 1e-20:
        return np.eye(4, dtype=np.float64) * np.array([-1.,1.,1.,1.]) # If weights are too small assume Minkowski
    g_eval = np.zeros((4,4), dtype=np.float64)
    for i in range(10):
        s = 0.
        for j in range(k):
            idx = neighbors[j]
            s += weights[j] * g_grid[idx,j]
        mu, nu = pairs[i]
        g_eval[mu,nu] = s / sum_w
        if mu != nu: g_eval[nu,mu] = s / sum_w
    return g_eval

@njit
def Gamma_interp(x:np.ndarray, grid:np.ndarray, Gamma_grid:np.ndarray, neighbors:np.ndarray,
                 k:int=12, power:float=2.0, eps:float=1e-12):
    """Returns the interpolated metric at x. Call only after grid points are saved to file.
    
    x: A point in spacetime described in metric coordinates.
    
    g_grid: The metric tensor (N, 10) evaluated on a selection of grid points.
    
    neighbors: np.ndarray with shape (N, k) specifying the index (in grid) of the k-th closest neighbors to a given point.
    Make sure that no direction dominates the closest neighbors.
    
    power: Describes how quickly the weights of neighboring points fall off with distance."""

    assert k <= neighbors.shape[1]
    pairs = np.array([(0,0,0),(0,0,1),(0,0,2),(0,0,3),(0,1,1),(0,1,2),(0,1,3),(0,2,2),(0,2,3),(0,3,3),
                      (1,0,0),(1,0,1),(1,0,2),(1,0,3),(1,1,1),(1,1,2),(1,1,3),(1,2,2),(1,2,3),(1,3,3),
                      (2,0,0),(2,0,1),(2,0,2),(2,0,3),(2,1,1),(2,1,2),(2,1,3),(2,2,2),(2,2,3),(2,3,3),
                      (3,0,0),(3,0,1),(3,0,2),(3,0,3),(3,1,1),(3,1,2),(3,1,3),(3,2,2),(3,2,3),(3,3,3)])
    dists = np.empty(k, dtype=np.float64)
    weights = np.empty(k, dtype=np.float64)

    for i in range(k):
        idx = neighbors[i]
        dp = grid[idx] - x
        d = np.sqrt(np.dot(dp, dp) + eps)
        dists[i] = d
        weights[i] = 1/(d**power) if d > 1e-20 else 1e20

    sum_w = np.sum(weights)
    if sum_w < 1e-20: return np.zeros((4,4,4), dtype=np.float64)
    Gamma_eval = np.zeros((4,4,4), dtype=np.float64)
    for i in range(40):
        s = 0.
        for j in range(k):
            idx = neighbors[j]
            s += weights[j] * Gamma_grid[idx,j]
        c, a, b = pairs[i]
        Gamma_eval[c,a,b] = s / sum_w
        if a != b: Gamma_eval[c,b,a] = s / sum_w
    return Gamma_eval

@njit(fastmath=True, nogil=True, cache=True)
def geodesic_eq(_, y, grid, Gamma_grid, neighbors):
    x, v = y[:4], y[4:]
    A = np.zeros(4)
    chr_syms = Gamma_interp(x, grid, Gamma_grid, neighbors)
    for c in range(4):
        for a in range(4):
            for b in range(4):
                A[c] -= chr_syms[c,a,b] * v[a] * v[b]
    return np.array([v[0], v[1], v[2], v[3], A[0], A[1], A[2], A[3]])

@njit
def singularity(y, grid, g_grid, neighbors):
    """Returns the largest-valued entry of the metric evaluated at y, subtracted by 1e10.

        For detecting singularities where the metric blows up."""
    
    x = y[:4]
    g, g_eval = np.abs(metric_interp(x, grid, g_grid, neighbors)), []
    for i in range(4):
        for j in range(4):
            g_eval.append(g[i,j])
    return max(g_eval)
            
@njit
def coord_pos(pos:np.ndarray) -> np.ndarray:
    """pos: A spacetime event described in Minkowski coordinates (relative to a stationary observer at infinity).

    Performs a coordinate transformation from Minkowski coordinates to that of the gravitational field's metric."""

    return np.array([X0(pos[0], pos[1], pos[2], pos[3]),
                     X1(pos[0], pos[1], pos[2], pos[3]),
                     X2(pos[0], pos[1], pos[2], pos[3]),
                     X3(pos[0], pos[1], pos[2], pos[3])], dtype=float64)

@njit
def mink_pos(pos:np.ndarray) -> np.ndarray:
    """pos: A spacetime event described in the coordiantes of the metric (relative to a stationary observer at infinity).

    Performs a coordinate transformation from the gravitational field's metric to Minkowski coordinates."""

    return np.array([T(pos[0], pos[1], pos[2], pos[3]),
                     X(pos[0], pos[1], pos[2], pos[3]),
                     Y(pos[0], pos[1], pos[2], pos[3]),
                     Z(pos[0], pos[1], pos[2], pos[3])], dtype=float64)

@njit
def coord_vel(vec:np.ndarray, pos:np.ndarray) -> np.ndarray:
    """vec: A contravariant four-vector described in Minkowski coordinates (relative to a stationary observer at infinity).

    pos: A spacetime point described in Minkowski coordinates (relative to a stationary observer at infinity).

    Performs a contravariant transformation of vec at pos from Minkowski coordinates to that of the gravitational field's metric."""

    j0 = [J00(pos[0], pos[1], pos[2], pos[3]), J01(pos[0], pos[1], pos[2], pos[3]), J02(pos[0], pos[1], pos[2], pos[3]), J03(pos[0], pos[1], pos[2], pos[3])]
    j1 = [J10(pos[0], pos[1], pos[2], pos[3]), J11(pos[0], pos[1], pos[2], pos[3]), J12(pos[0], pos[1], pos[2], pos[3]), J13(pos[0], pos[1], pos[2], pos[3])]
    j2 = [J20(pos[0], pos[1], pos[2], pos[3]), J21(pos[0], pos[1], pos[2], pos[3]), J22(pos[0], pos[1], pos[2], pos[3]), J23(pos[0], pos[1], pos[2], pos[3])]
    j3 = [J30(pos[0], pos[1], pos[2], pos[3]), J31(pos[0], pos[1], pos[2], pos[3]), J32(pos[0], pos[1], pos[2], pos[3]), J33(pos[0], pos[1], pos[2], pos[3])]

    new_vel = [j0[0]*vec[0] + j0[1]*vec[1] + j0[2]*vec[2] + j0[3]*vec[3],
               j1[0]*vec[0] + j1[1]*vec[1] + j1[2]*vec[2] + j1[3]*vec[3],
               j2[0]*vec[0] + j2[1]*vec[1] + j2[2]*vec[2] + j2[3]*vec[3],
               j3[0]*vec[0] + j3[1]*vec[1] + j3[2]*vec[2] + j3[3]*vec[3]]
    return np.array(new_vel, dtype=float64)
    
@njit
def mink_vel(vec:np.ndarray, pos:np.ndarray) -> np.ndarray:
    """vec: A contravariant four-vector described in metric coordinates (relative to a stationary observer at infinity).

    pos: A spacetime point described in metric coordinates (relative to a stationary observer at infinity).

    Performs a contravariant transformation of vec at pos from metric coordinates to Minkowski coordinates."""

    j0 = [Jinv00(pos[0], pos[1], pos[2], pos[3]), Jinv01(pos[0], pos[1], pos[2], pos[3]), Jinv02(pos[0], pos[1], pos[2], pos[3]), Jinv03(pos[0], pos[1], pos[2], pos[3])]
    j1 = [Jinv10(pos[0], pos[1], pos[2], pos[3]), Jinv11(pos[0], pos[1], pos[2], pos[3]), Jinv12(pos[0], pos[1], pos[2], pos[3]), Jinv13(pos[0], pos[1], pos[2], pos[3])]
    j2 = [Jinv20(pos[0], pos[1], pos[2], pos[3]), Jinv21(pos[0], pos[1], pos[2], pos[3]), Jinv22(pos[0], pos[1], pos[2], pos[3]), Jinv23(pos[0], pos[1], pos[2], pos[3])]
    j3 = [Jinv30(pos[0], pos[1], pos[2], pos[3]), Jinv31(pos[0], pos[1], pos[2], pos[3]), Jinv32(pos[0], pos[1], pos[2], pos[3]), Jinv33(pos[0], pos[1], pos[2], pos[3])]

    new_vel = [j0[0]*vec[0] + j0[1]*vec[1] + j0[2]*vec[2] + j0[3]*vec[3],
               j1[0]*vec[0] + j1[1]*vec[1] + j1[2]*vec[2] + j1[3]*vec[3],
               j2[0]*vec[0] + j2[1]*vec[1] + j2[2]*vec[2] + j2[3]*vec[3],
               j3[0]*vec[0] + j3[1]*vec[1] + j3[2]*vec[2] + j3[3]*vec[3]]
    return np.array(new_vel, dtype=float64)

@njit
def normed(vec:Vec, pos:np.ndarray) -> np.ndarray:
    """vec: A contravariant four-vector described in Minkowski coordinates (relative to a stationary observer at infinity).

    pos: A spacetime point described in Minkowski coordinates (relative to a stationary observer at infinity).
    
    Returns the normed vector (norm=0) at pos in metric coordinates whose spatial components are the transformed components of vec."""

    a = null_cond_a(pos[0], pos[1], pos[2], pos[3], vec.x, vec.y, vec.z)
    b = null_cond_b(pos[0], pos[1], pos[2], pos[3], vec.x, vec.y, vec.z)
    c = null_cond_c(pos[0], pos[1], pos[2], pos[3], vec.x, vec.y, vec.z)
    disc = b**2 - 4*a*c
    v0 = min((-b-np.sqrt(disc))/(2*a), (-b+np.sqrt(disc))/(2*a))
    new_vel = np.array([v0, vec.x, vec.y, vec.z])
    return coord_vel(new_vel, pos)
