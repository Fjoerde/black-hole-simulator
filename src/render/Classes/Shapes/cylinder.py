import numpy as np
from numba import njit

from render.Classes.base import *
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

@njit
def init(self, radius:float, height:float):
    self.radius = radius
    self.height = height
    if self.radius <= 0 or self.height <= 0: raise ValueError("Cylinder has non-positive radius or height.")

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
