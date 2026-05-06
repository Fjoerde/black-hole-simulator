from numba import njit
from render.Classes.math import *

@njit
def init(self): pass

@njit
def spec_int(self, pt):
    return Function(Grid(Patch([np.zeros(1, dtype=np.float64)])), np.zeros((1,1), dtype=np.float64))