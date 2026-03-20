import numpy as np
from numba import njit

from render.Classes.base import *
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

@njit
def bounding_box(self):
    u, v, w = (Vec(1,0,0)-Vec(1,0,0).dot(self.Z)*self.Z).normal(), (Vec(0,1,0)-Vec(0,1,0).dot(self.Z)*self.Z).normal(), (Vec(0,0,1)-Vec(0,0,1).dot(self.Z)*self.Z).normal()
    extrema = [self.c_top+self.radius*u, self.c_top+self.radius*v, self.c_top+self.radius*w,
                self.c_top-self.radius*u, self.c_top-self.radius*v, self.c_top-self.radius*w,
                self.c_base+self.radius*u, self.c_base+self.radius*v, self.c_base+self.radius*w,
                self.c_base-self.radius*u, self.c_base-self.radius*v, self.c_base-self.radius*w]
    x_lst, y_lst, z_lst = [vec.x for vec in extrema], [vec.y for vec in extrema], [vec.z for vec in extrema]
    min_x, max_x = min(x_lst), max(x_lst)
    min_y, max_y = min(y_lst), max(y_lst)
    min_z, max_z = min(z_lst), max(z_lst)
    return BoundingBox(Vec(min_x, min_y, min_z), Vec(max_x, max_y, max_z))

@njit
def init(self, radius:float, height:float):
    self.radius = radius
    self.height = height
    if self.radius <= 0 or self.height <= 0: raise ValueError("Cylinder has non-positive radius or height.")
    self.c_top = self.pos+(self.height/2)*self.Z
    self.c_base = self.pos-(self.height/2)*self.Z
    self.bb = bounding_box(self)

@njit
def closest_pt(self, pt:Vec):
    cp = pt - self.pos
    s, phi, z = np.sqrt(cp.dot(self.X)**2+cp.dot(self.Y)**2), np.arctan2(cp.dot(self.Y), cp.dot(self.X)), cp.dot(self.Z)
    S = np.cos(phi)*self.X + np.sin(phi)*self.Y

    if s <= self.radius: base, top = pt-(z+self.height/2)*self.Z, pt-(z-self.height/2)*self.Z
    else: base, top = pt-(s-self.radius)*S-(z+self.height/2)*self.Z, pt-(s-self.radius)*S-(z-self.height/2)*self.Z
    if np.abs(z) <= self.height/2: side = pt-(s-self.radius)*S
    elif z > self.height/2: side = pt-(s-self.radius)*S-(z-self.height/2)*self.Z
    else: side = pt-(s-self.radius)*S-(z+self.height/2)*self.Z

    surface_lst = [base, top, side]
    dists = []
    for i in surface_lst: dists.append((i-pt).length())
    return surface_lst[dists.index(min(dists))]

@njit
def in_shape(self, pt:Vec):
    cp = pt - self.pos
    s, z = (cp - cp.dot(self.Z)*self.Z).length(), cp.dot(self.Z)
    return s < self.radius and np.abs(z) < self.height/2

@njit
def projection(self, pt:Vec):
    cp = (pt - self.pos).normal()
    theta, phi = np.arccos(cp.dot(self.Z)), np.arctan2(cp.dot(self.Y), cp.dot(self.X))
    u, v = theta/np.pi, (phi+np.pi)/(2*np.pi) 
    return u, v
