import numpy as np
from numba import njit
from numba.typed import List

from Classes.math import Vec

@njit
def init(self, grav_field):
    self.grav_field = grav_field

@njit
def derivative(self, _, y:np.ndarray) -> np.ndarray:
    y = np.ascontiguousarray(y)
    x, v = y[:4], y[4:]
    chr_syms = self.grav_field.sample_Gamma(x)
    A = np.zeros(4, dtype=np.float64)
    for c in range(4): A[c] -= (chr_syms[c] @ v) @ v # Geodesic equation
    y_new = np.concatenate((v, A))
    return np.ascontiguousarray(y_new)

@njit
def term_cond(self, _, y:np.ndarray, h:float) -> bool:
    if h < 1e-4:
        if np.max(self.grav_field.sample_g(y[:4])) > 200: return True
    else: return False

@njit
def max_step(self, t:float, y:np.ndarray, _) -> float:
    return np.inf
