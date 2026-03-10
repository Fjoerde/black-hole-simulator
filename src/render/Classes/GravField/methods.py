import numpy as np
from numba import njit, prange

from Classes.base import *
from Classes.GravField.funcs import *
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))


# Numba-jitted functions
@njit
def metric_at(x:np.ndarray):
    """Returns the metric tensor evaluated at x.
    
    x: A point in spacetime described in metric coordinates."""

    g = np.array([[g00(x[0], x[1], x[2], x[3]), g01(x[0], x[1], x[2], x[3]), g02(x[0], x[1], x[2], x[3]), g03(x[0], x[1], x[2], x[3])],
                  [g10(x[0], x[1], x[2], x[3]), g11(x[0], x[1], x[2], x[3]), g12(x[0], x[1], x[2], x[3]), g13(x[0], x[1], x[2], x[3])],
                  [g20(x[0], x[1], x[2], x[3]), g21(x[0], x[1], x[2], x[3]), g22(x[0], x[1], x[2], x[3]), g23(x[0], x[1], x[2], x[3])],
                  [g30(x[0], x[1], x[2], x[3]), g31(x[0], x[1], x[2], x[3]), g32(x[0], x[1], x[2], x[3]), g33(x[0], x[1], x[2], x[3])]], dtype=float64)
    return g

@njit(fastmath=True)
def chr_sym(x:np.ndarray, h:float=1e-7):
    """Returns the Christoffel symbol evaluated at x.
    
    x: A point in spacetime described in metric coordinates."""

    g = metric_at(x)
    g_inv = np.linalg.inv(g)
    dg = np.zeros((4,4,4)) # dg[k,i,j] = Derivative of g_ij with respect to x^k
    for k in range(4):
        xp = x.copy(); xp[k] += h
        xm = x.copy(); xm[k] -= h
        gp, gm = metric_at(xp), metric_at(xm)
        dg[k] = (gp - gm) / (h * 2)
    
    chr_syms = np.zeros((4,4,4), dtype=float64)
    for c in range(4):
        for a in range(4):
            for b in range(4):
                s = 0
                for d in range(4):
                    s += g_inv[c,d] * (dg[a,b,d] + dg[b,a,d] - dg[d,a,b])
                chr_syms[c,a,b] = s / 2
    return chr_syms

@njit(fastmath=True)
def geodesic_eq(t, y):
    x, v = y[:4], y[4:]
    A = np.zeros(4)
    
    Gammas = chr_sym(x)
    for c in range(4):
        for a in range(4):
            for b in range(4):
                A[c] -= Gammas[c,a,b] * v[a] * v[b]

    return np.array([v[0], v[1], v[2], v[3], A[0], A[1], A[2], A[3]])

"""
@njit(parallel=True, fastmath=True, cache=True)
def kretsch_scal(x:np.ndarray, h:float=1e-7):
    ""x: A spacetime point described in metric coordinates.

    Returns the Kretschmann scalar of the metric at x.""
    Gammas = chr_sym(x)
    dGammas = np.zeros((4,4,4,4)) # dGammas[d,c,a,b] = Derivative of Gamma[c,a,b] with respect to x^d
    for d in range(4):
        xp = x.copy(); xp[d] += h
        xm = x.copy(); xm[d] -= h
        Gp, Gm = chr_sym(xp), chr_sym(xm)
        dGammas[d] = (Gp - Gm) / (h * 2)
    
    riem = np.zeros((4,4,4,4))
    for d in prange(4):
        for c in prange(4):
            for a in prange(4):
                for b in prange(4):
                    s = dGammas[a,d,c,b] - dGammas[b,d,c,a]
                    for e in prange(4):
                        s += Gammas[e,c,d] * Gammas[d,e,b]
                        s -= Gammas[e,b,c] * Gammas[d,e,a]
                    riem[d,c,a,b] = s
    
    K = 0.
    for d in prange(4):
        for c in prange(4):
            for b in prange(4):
                for a in prange(4):
                    K += riem[d,c,a,b]**2
    return K
"""

@njit
def singularity(y):
    """Returns the largest-valued entry of the metric evaluated at y, subtracted by 1e10.

        For detecting singularities where the metric blows up."""
    
    x = y[:4]
    g, g_eval = np.abs(metric_at(x)), []
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
