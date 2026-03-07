from numba import njit

from render.Classes.base import *

@njit
def init(self, col1:Vec, col2:Vec):
    self.col1 = col1
    self.col2 = col2

@njit
def sample(self, u, v):
    grid_u, grid_v = int(10*u), int(10*v)
    col_num = (grid_u + grid_v) % 2
    if col_num == 0: return self.col1
    elif col_num == 1: return self.col2
