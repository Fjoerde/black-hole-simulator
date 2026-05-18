import numpy as np
from numba import njit

from Classes.math import *

@njit
def init(self, temp, col_converter):
    self.temp = temp
    self.specint_grid = col_converter.grid

@njit
def spec_int(self, pt):
    if not self.shape.on_surface(pt): raise ValueError("Expected a point on the surface of the Shape.")
    
    grid = self.specint_grid
    wvls = grid.pts.reshape(len(grid.pts)) * 1e-9

    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23
    vals = 2*h*c**2/wvls**5 * 1/(np.exp((h*c)/(wvls*k*self.temp)) - 1) * 1e-9
    return Function(grid, vals.reshape(len(vals), 1))
