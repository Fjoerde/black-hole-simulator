import numpy as np
from numba import njit

from render.Classes.base import *
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

@njit
def bounding_box(self):
    u, v, w = (Vec(1,0,0)-Vec(1,0,0).dot(self.Z)*self.Z).normal(), (Vec(0,1,0)-Vec(0,1,0).dot(self.Z)*self.Z).normal(), (Vec(0,0,1)-Vec(0,0,1).dot(self.Z)*self.Z).normal()
    extrema = [self.c_top+self.r_out*u, self.c_top+self.r_out*v, self.c_top+self.r_out*w,
                self.c_top-self.r_out*u, self.c_top-self.r_out*v, self.c_top-self.r_out*w,
                self.c_base+self.r_out*u, self.c_base+self.r_out*v, self.c_base+self.r_out*w,
                self.c_base-self.r_out*u, self.c_base-self.r_out*v, self.c_base-self.r_out*w]

    x_lst, y_lst, z_lst = [vec.x for vec in extrema], [vec.y for vec in extrema], [vec.z for vec in extrema]
    min_x, max_x = min(x_lst), max(x_lst)
    min_y, max_y = min(y_lst), max(y_lst)
    min_z, max_z = min(z_lst), max(z_lst)
    return BoundingBox(Vec(min_x, min_y, min_z), Vec(max_x, max_y, max_z))

@njit
def init(self, r_in:float, r_out:float, height:float):
    self.r_in = r_in
    self.r_out = r_out
    self.height = height
    if self.r_in <= 0 or self.r_out <= 0 or self.height <= 0: raise ValueError("Annulus has non-positive radii or height.")
    if self.r_in >= self.r_out: raise ValueError("r_in must be smaller than r_out.")

    self.c_top = self.pos+(self.height/2)*self.Z
    self.c_base = self.pos-(self.height/2)*self.Z
    self.bb = bounding_box(self)

@njit
def closest_pt(self, pt):
    cp = pt - self.pos
    s, phi, z = np.sqrt(cp.dot(self.X)**2+cp.dot(self.Y)**2), np.arctan2(cp.dot(self.Y), cp.dot(self.X)), cp.dot(self.Z)
    S = np.cos(phi)*self.X + np.sin(phi)*self.Y

    if self.r_in <= s <= self.r_out: base, top = pt-(z+self.height/2)*self.Z, pt-(z-self.height/2)*self.Z
    elif s > self.r_out: base, top = pt-(s-self.r_out)*S-(z+self.height/2)*self.Z, pt-(s-self.r_out)*S-(z-self.height/2)*self.Z
    else: base, top = pt-(s-self.r_in)*S-(z+self.height/2)*self.Z, pt-(s-self.r_in)*S-(z-self.height/2)*self.Z
    if np.abs(z) <= self.height/2: side_in, side_out = pt-(s-self.r_in)*S, pt-(s-self.r_out)*S
    elif z > self.height/2: side_in, side_out = pt-(s-self.r_in)*S-(z-self.height/2)*self.Z, pt-(s-self.r_out)*S-(z-self.height/2)*self.Z
    else: side_in, side_out = pt-(s-self.r_in)*S-(z+self.height/2)*self.Z, pt-(s-self.r_out)*S-(z+self.height/2)*self.Z

    surface_lst = [base, top, side_in, side_out]
    dists = []
    for i in surface_lst: dists.append((i-pt).length())
    return surface_lst[dists.index(min(dists))]

@njit    
def in_shape(self, pt):
    cp = pt - self.pos
    s, z = (cp - cp.dot(self.Z)*self.Z).length(), cp.dot(self.Z)
    return self.r_in < s < self.r_out and np.abs(z) < self.height/2

@njit    
def projection(self, pt):
    cp1 = pt - self.pos
    phi = np.arctan2(cp1.dot(self.Y), cp1.dot(self.X)) # Toroidal coordinate
    S = np.cos(phi)*self.X + np.sin(phi)*self.Y
    cp2 = pt - (self.pos + (self.r_in+self.r_out)/2 * S)
    theta = np.arctan2(cp2.dot(self.Z), cp2.dot(S)) # Poloidal coordinate
    u, v = (phi+np.pi)/(2*np.pi), (theta+np.pi)/(2*np.pi) 
    return u, v

@njit
def straight_hit(self, ray_pos, ray_dir): # Reused a lot of code from Cylinder
    ray_dir.is_normal()
    if not self.bb.hit_bb(ray_pos, ray_dir): return np.inf

    top_k, base_k = self.c_top.dot(self.Z), self.c_base.dot(-self.Z)
    if np.abs(ray_dir.dot(self.Z)) < 1e-10: t_top, t_base = 1e10, 1e10
    else: t_top, t_base = (top_k-ray_pos.dot(self.Z))/(ray_dir.dot(self.Z)), (base_k-ray_pos.dot(-self.Z))/(ray_dir.dot(-self.Z))
    hit_top, hit_base = ray_pos+t_top*ray_dir, ray_pos+t_base*ray_dir
    if not self.r_in <= (hit_top - self.c_top).length() <= self.r_out: t_top = np.inf 
    if not self.r_in <= (hit_base - self.c_base).length() <= self.r_out: t_base = np.inf 
    # Outer surface
    pp = ray_pos - self.pos
    a = 1 - ray_dir.dot(self.Z)**2
    b = 2 * (ray_dir.dot(pp) - ray_dir.dot(self.Z)*pp.dot(self.Z))
    c = pp.length()**2 - pp.dot(self.Z)**2 - self.r_in**2
    disc = b**2 - 4*a*c
    if disc >= 0:
        t_side_in = (-b + np.sqrt(disc)) / (2 * a)
        hit_side_in = ray_pos + t_side_in*ray_dir
        if np.abs((hit_side_in - self.pos).dot(self.Z)) > self.height/2: t_side_in = np.inf
    else: t_side_in = np.inf
    # Inner surface
    c += self.r_in**2 - self.r_out**2
    disc = b**2 - 4*a*c
    if disc >= 0:
        t_side_out = (-b - np.sqrt(disc)) / (2 * a)
        hit_side_out = ray_pos + t_side_out*ray_dir
        if np.abs((hit_side_out - self.pos).dot(self.Z)) > self.height/2: t_side_out = np.inf
    else: t_side_out = np.inf

    all_inf = True
    for t in [t_top, t_base, t_side_in, t_side_out]:
        if t != np.inf: all_inf = False
    if all_inf: return np.inf
    else: return min([t_top, t_base, t_side_in, t_side_out])
