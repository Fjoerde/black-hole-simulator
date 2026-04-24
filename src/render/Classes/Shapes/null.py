import numpy as np
from numba import njit

from render.Classes.base import *
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

@njit
def init(self): pass

@njit
def closest_pt(self, pt:Vec): return Vec(np.inf,np.inf,np.inf)

@njit
def in_shape(self, pt:Vec): return False
    
@njit
def projection(self, pt:Vec): return 0, 0
