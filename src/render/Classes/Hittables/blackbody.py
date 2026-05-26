import numpy as np
from numba import njit

from Classes.math import *

@njit
def init(self, temp, col_converter):
    self.temp = temp

    wvls = col_converter.grid.pts.flatten() * 1e-9
    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23
    vals = 2*h*c**2/wvls**5 * 1/(np.exp((h*c)/(wvls*k*self.temp)) - 1) * 1e-9
    spec_int = Function(col_converter.grid, np.ascontiguousarray(vals.reshape(len(vals), 1)))
    self.col = col_converter.get_rgb(spec_int)

@njit
def get_col(self, pt): return self.col
