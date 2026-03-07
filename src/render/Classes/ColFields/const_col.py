from numba import njit

from Classes.base import *

@njit
def init(self, col:Vec):
    self.col = col

@njit
def sample(self, _, __):
    return self.col
