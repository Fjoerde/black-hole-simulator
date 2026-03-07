from numba import njit
from render.Classes.base import *

@njit
def init(self): pass

@njit
def color(self, pt): return Vec(0,0,0)