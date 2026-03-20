import numpy as np
from numba import njit

@njit
def init(self): pass

@njit
def sample_g(self, x): return np.diag(np.array([-1., 1., 1., 1.], dtype=np.float64))

@njit
def sample_Gamma(self, x): return np.zeros((4,4,4), dtype=np.float64)

@njit
def coord_pos(self, x):
    T, X, Y, Z = x
    out = np.zeros(4, dtype=np.float64)
    out[0] = T; out[1] = X; out[2] = Y; out[3] = Z
    return out

@njit
def mink_pos(self, x):
    T, X, Y, Z = x
    out = np.zeros(4, dtype=np.float64)
    out[0] = T; out[1] = X; out[2] = Y; out[3] = Z
    return out

@njit
def jacobian(self, x): return np.eye(4, dtype=np.float64)

@njit
def jacobian_inv(self, x): return np.eye(4, dtype=np.float64)
