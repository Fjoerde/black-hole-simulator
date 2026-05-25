from numba import njit
from Classes.math import *

@njit
def init(self): pass

@njit
def get_col(self, pt):
    return np.zeros(3, dtype=np.float64)