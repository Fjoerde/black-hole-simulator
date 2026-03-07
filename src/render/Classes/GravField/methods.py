import numpy as np
from numba import njit

from render.Classes.base import *
from render.Classes.GravField.funcs import *
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))


# Numba-jitted functions
@njit
def geodesic_eq(t, y):
    x, v = y[:4], y[4:]
    A = np.zeros(4)
    
    A[0] -= Gamma000(x[0], x[1], x[2], x[3]) * v[0] * v[0]
    A[0] -= Gamma001(x[0], x[1], x[2], x[3]) * v[0] * v[1]
    A[0] -= Gamma002(x[0], x[1], x[2], x[3]) * v[0] * v[2]
    A[0] -= Gamma003(x[0], x[1], x[2], x[3]) * v[0] * v[3]
    A[0] -= Gamma010(x[0], x[1], x[2], x[3]) * v[1] * v[0]
    A[0] -= Gamma011(x[0], x[1], x[2], x[3]) * v[1] * v[1]
    A[0] -= Gamma012(x[0], x[1], x[2], x[3]) * v[1] * v[2]
    A[0] -= Gamma013(x[0], x[1], x[2], x[3]) * v[1] * v[3]
    A[0] -= Gamma020(x[0], x[1], x[2], x[3]) * v[2] * v[0]
    A[0] -= Gamma021(x[0], x[1], x[2], x[3]) * v[2] * v[1]
    A[0] -= Gamma022(x[0], x[1], x[2], x[3]) * v[2] * v[2]
    A[0] -= Gamma023(x[0], x[1], x[2], x[3]) * v[2] * v[3]
    A[0] -= Gamma030(x[0], x[1], x[2], x[3]) * v[3] * v[0]
    A[0] -= Gamma031(x[0], x[1], x[2], x[3]) * v[3] * v[1]
    A[0] -= Gamma032(x[0], x[1], x[2], x[3]) * v[3] * v[2]
    A[0] -= Gamma033(x[0], x[1], x[2], x[3]) * v[3] * v[3]

    A[1] -= Gamma100(x[0], x[1], x[2], x[3]) * v[0] * v[0]
    A[1] -= Gamma101(x[0], x[1], x[2], x[3]) * v[0] * v[1]
    A[1] -= Gamma102(x[0], x[1], x[2], x[3]) * v[0] * v[2]
    A[1] -= Gamma103(x[0], x[1], x[2], x[3]) * v[0] * v[3]
    A[1] -= Gamma110(x[0], x[1], x[2], x[3]) * v[1] * v[0]
    A[1] -= Gamma111(x[0], x[1], x[2], x[3]) * v[1] * v[1]
    A[1] -= Gamma112(x[0], x[1], x[2], x[3]) * v[1] * v[2]
    A[1] -= Gamma113(x[0], x[1], x[2], x[3]) * v[1] * v[3]
    A[1] -= Gamma120(x[0], x[1], x[2], x[3]) * v[2] * v[0]
    A[1] -= Gamma121(x[0], x[1], x[2], x[3]) * v[2] * v[1]
    A[1] -= Gamma122(x[0], x[1], x[2], x[3]) * v[2] * v[2]
    A[1] -= Gamma123(x[0], x[1], x[2], x[3]) * v[2] * v[3]
    A[1] -= Gamma130(x[0], x[1], x[2], x[3]) * v[3] * v[0]
    A[1] -= Gamma131(x[0], x[1], x[2], x[3]) * v[3] * v[1]
    A[1] -= Gamma132(x[0], x[1], x[2], x[3]) * v[3] * v[2]
    A[1] -= Gamma133(x[0], x[1], x[2], x[3]) * v[3] * v[3]

    A[2] -= Gamma200(x[0], x[1], x[2], x[3]) * v[0] * v[0]
    A[2] -= Gamma201(x[0], x[1], x[2], x[3]) * v[0] * v[1]
    A[2] -= Gamma202(x[0], x[1], x[2], x[3]) * v[0] * v[2]
    A[2] -= Gamma203(x[0], x[1], x[2], x[3]) * v[0] * v[3]
    A[2] -= Gamma210(x[0], x[1], x[2], x[3]) * v[1] * v[0]
    A[2] -= Gamma211(x[0], x[1], x[2], x[3]) * v[1] * v[1]
    A[2] -= Gamma212(x[0], x[1], x[2], x[3]) * v[1] * v[2]
    A[2] -= Gamma213(x[0], x[1], x[2], x[3]) * v[1] * v[3]
    A[2] -= Gamma220(x[0], x[1], x[2], x[3]) * v[2] * v[0]
    A[2] -= Gamma221(x[0], x[1], x[2], x[3]) * v[2] * v[1]
    A[2] -= Gamma222(x[0], x[1], x[2], x[3]) * v[2] * v[2]
    A[2] -= Gamma223(x[0], x[1], x[2], x[3]) * v[2] * v[3]
    A[2] -= Gamma230(x[0], x[1], x[2], x[3]) * v[3] * v[0]
    A[2] -= Gamma231(x[0], x[1], x[2], x[3]) * v[3] * v[1]
    A[2] -= Gamma232(x[0], x[1], x[2], x[3]) * v[3] * v[2]
    A[2] -= Gamma233(x[0], x[1], x[2], x[3]) * v[3] * v[3]

    A[3] -= Gamma300(x[0], x[1], x[2], x[3]) * v[0] * v[0]
    A[3] -= Gamma301(x[0], x[1], x[2], x[3]) * v[0] * v[1]
    A[3] -= Gamma302(x[0], x[1], x[2], x[3]) * v[0] * v[2]
    A[3] -= Gamma303(x[0], x[1], x[2], x[3]) * v[0] * v[3]
    A[3] -= Gamma310(x[0], x[1], x[2], x[3]) * v[1] * v[0]
    A[3] -= Gamma311(x[0], x[1], x[2], x[3]) * v[1] * v[1]
    A[3] -= Gamma312(x[0], x[1], x[2], x[3]) * v[1] * v[2]
    A[3] -= Gamma313(x[0], x[1], x[2], x[3]) * v[1] * v[3]
    A[3] -= Gamma320(x[0], x[1], x[2], x[3]) * v[2] * v[0]
    A[3] -= Gamma321(x[0], x[1], x[2], x[3]) * v[2] * v[1]
    A[3] -= Gamma322(x[0], x[1], x[2], x[3]) * v[2] * v[2]
    A[3] -= Gamma323(x[0], x[1], x[2], x[3]) * v[2] * v[3]
    A[3] -= Gamma330(x[0], x[1], x[2], x[3]) * v[3] * v[0]
    A[3] -= Gamma331(x[0], x[1], x[2], x[3]) * v[3] * v[1]
    A[3] -= Gamma332(x[0], x[1], x[2], x[3]) * v[3] * v[2]
    A[3] -= Gamma333(x[0], x[1], x[2], x[3]) * v[3] * v[3]

    return np.array([v[0], v[1], v[2], v[3], A[0], A[1], A[2], A[3]])

@njit
def singularity(y):
    """Returns the largest-valued entry of the metric evaluated at y, subtracted by 1e10.

        For detecting singularities where the metric blows up."""
    
    x = y[:4]
    g_eval = [g00(x[0], x[1], x[2], x[3]), g01(x[0], x[1], x[2], x[3]), g02(x[0], x[1], x[2], x[3]), g03(x[0], x[1], x[2], x[3]),
              g10(x[0], x[1], x[2], x[3]), g11(x[0], x[1], x[2], x[3]), g12(x[0], x[1], x[2], x[3]), g13(x[0], x[1], x[2], x[3]),
              g20(x[0], x[1], x[2], x[3]), g21(x[0], x[1], x[2], x[3]), g22(x[0], x[1], x[2], x[3]), g23(x[0], x[1], x[2], x[3]),
              g30(x[0], x[1], x[2], x[3]), g31(x[0], x[1], x[2], x[3]), g32(x[0], x[1], x[2], x[3]), g33(x[0], x[1], x[2], x[3])]
    return max(g_eval) - 1e10
            
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
def kretsch_scal(pos:np.ndarray) -> float:
    """pos: A spacetime point described in metric coordinates.

    Returns the Kretschmann scalar of the metric at pos."""

    return K(pos[0], pos[1], pos[2], pos[3])

@njit
def normed(vec:Vec, pos:np.ndarray) -> np.ndarray:
    """vec: A contravariant four-vector described in Minkowski coordinates (relative to a stationary observer at infinity).

    pos: A spacetime point described in Minkowski coordinates (relative to a stationary observer at infinity).
    
    Returns the normed vector (norm=0) at pos in metric coordinates whose spatial components are the transformed components of vec."""

    a = null_cond_a(pos[0], pos[1], pos[2], pos[3], vec.x, vec.y, vec.z)
    b = null_cond_b(pos[0], pos[1], pos[2], pos[3], vec.x, vec.y, vec.z)
    c = null_cond_c(pos[0], pos[1], pos[2], pos[3], vec.x, vec.y, vec.z)
    v0 = (-b - np.sqrt(b**2 - 4*a*c)) / (2 * a)
    new_vel = np.array([v0, vec.x, vec.y, vec.z])
    return coord_vel(new_vel, pos)
