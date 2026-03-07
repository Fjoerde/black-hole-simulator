import numpy as np
from numba import njit

from Classes.base import *
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

@njit
def bounding_box(self): return BoundingBox(Vec(np.inf,np.inf,np.inf), Vec(np.inf,np.inf,np.inf))

@njit
def init(self):
    self.bb = bounding_box(self)

@njit
def closest_pt(self, pt:Vec): return Vec(np.inf,np.inf,np.inf)

@njit
def in_shape(self, pt:Vec): return False
    
@njit
def projection(self, pt:Vec): return 0, 0

@njit
def straight_hit(self, ray_pos:Vec, ray_dir:Vec): return np.inf