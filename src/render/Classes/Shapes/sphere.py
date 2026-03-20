import numpy as np
from numba import njit

from render.Classes.base import *
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

@njit
def bounding_box(self):
    min = self.pos - self.radius*Vec(1,1,1)
    max = self.pos + self.radius*Vec(1,1,1)
    return BoundingBox(min, max)

@njit
def init(self, radius:float):
    self.radius = radius
    if self.radius <= 0: raise ValueError("Sphere has non-positive radius.")
    self.bb = bounding_box(self)

@njit
def closest_pt(self, pt:Vec):
    normal = (pt - self.pos).normal()
    return self.pos + normal * self.radius

@njit
def in_shape(self, pt:Vec):
    return (pt - self.pos).length() < self.radius
    
@njit
def projection(self, pt:Vec):
    cp = (pt - self.pos).normal()
    theta, phi = np.arccos(cp.dot(self.Z)), np.arctan2(cp.dot(self.Y), cp.dot(self.X))
    u, v = theta/np.pi, (phi+np.pi)/(2*np.pi) 
    return u, v
