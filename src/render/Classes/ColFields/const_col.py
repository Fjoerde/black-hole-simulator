from numba import njit

from render.Classes.base import *

@njit
def init(self, col:Vec):
    self.col = col

@njit
def sample(self, _, __):
    return self.col
