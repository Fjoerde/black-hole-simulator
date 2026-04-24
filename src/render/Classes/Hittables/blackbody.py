import numpy as np
from numba import njit

from render.Classes.base import *

@njit
def init(self, temp):
    self.temp = temp

@njit
def spec_int(self, pt):
    if not self.shape.on_surface(pt): raise ValueError("Expected a point on the surface of the Shape.")
    
    h = 6.63e-34; c = 3e8; k = 1.38e-23
    w_peak = 2.898e-3 / self.temp
    w_min = 0.05 * w_peak; w_max = 10 * w_peak
    grid = Grid().add_patch(Patch([np.logspace(np.log10(w_min), np.log10(w_max), 81)]))
    wvls = grid.pts.reshape(len(grid.pts))
    vals = 2*h*c*2/wvls**5 * 1/(np.exp((h*c)/(wvls*k*self.temp)) - 1)
    return Function(grid, vals.reshape(len(vals), 1))
