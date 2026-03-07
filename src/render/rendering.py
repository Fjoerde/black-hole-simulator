import numpy as np
from PIL import Image
from numba import njit, prange
from numba_progress import ProgressBar

from Classes.base import *
from Classes.tags import *
import Classes.GravField.methods as GravMethod
from Classes.classes import RenderSettings

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Numba-jitted functions operate in no-python mode, which severely limits the
# compactness of the code since a lot of features in Python may not be implemented.

@njit
def rk4_step(t, y, h): # h is a step size
    k1 = GravMethod.geodesic_eq(t, y)
    k2 = GravMethod.geodesic_eq(t+h/2, y+h*k1/2)
    k3 = GravMethod.geodesic_eq(t+h/2, y+h*k2/2)
    k4 = GravMethod.geodesic_eq(t+h, y+h*k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

@njit
def integrator(settings:RenderSettings, y0:float, max_t:float, tol:float=1e-6, h_init:float=0.1):
    """Returns the point and the object the light ray hits."""
    t, y, h = 0., y0, h_init

    while t < max_t:

        # Adapt step size to how close the point is to any object
        deriv, speed = GravMethod.geodesic_eq(t, y), 0
        for i in deriv: speed += i ** 2
        speed = np.sqrt(speed)
        dists = []
        for obj in settings.scene: dists.append(obj.shape.in_shape_int(y))
        min_dist = np.inf
        for i in dists:
            if i < min_dist: min_dist = i
        if h > 1.5*min_dist/speed: h = 1.5*min_dist/speed

        # Perform integration
        if t + h > max_t: h = max_t - t # Prevents overstepping
        y_coarse = rk4_step(t, y, h) # Coarse step
        y_mid = rk4_step(t, y, h/2)
        y_fine = rk4_step(t+h/2, y_mid, h/2) # Fine step made of 2 substeps
        # Error estimate
        error = np.max(np.abs(y_coarse - y_fine)) / 15
        factor = (tol / error) ** (1/5)
        if error <= tol:
            t += h
            y = y_fine
            h *= 0.9 * factor # Safety factor of 0.9
        else: # Reject solution and reduce step size
            if factor > 0.1: h *= 0.9 * factor
            else: h *= 0.9 * 0.1

        # Check for any hits
        hit_obj, message = settings.scene[0], "background"
        for obj in settings.scene:
            if obj.shape.in_shape_int(y) >= 0.:
                hit_obj, message = obj, "hit object"
                break
        if message == "hit object": break
        if settings.escape(y) > 0:
            message = "background"
            break
        if GravMethod.singularity(y) > 0:
            message = "singularity"
            break
    
    return y, message, hit_obj

@njit
def trace(pos:Vec, dir:Vec, settings:RenderSettings): # Returns color the light ray starting from pos in dir 'sees'
    dir.is_normal()
    col = Vec(0,0,0)
    X0 = GravMethod.coord_pos(np.array([0, pos.x, pos.y, pos.z]))
    V0 = GravMethod.normed(dir, np.array([0, pos.x, pos.y, pos.z]))

    if settings.straight_approx(X0, V0):
        obj, hit, hit_pt, t = settings.scene[0], False, Vec(np.inf,np.inf,np.inf), np.inf
        hit_lst = []
        for obj_ in settings.scene:
            if obj_.shape.bb.hit_bb(pos, dir): hit_lst.append(obj_)
        for obj_ in hit_lst:
            t_ = obj_.shape.straight_hit(pos, dir)
            if t_ != np.inf and t_ < t:
                obj, hit, hit_pt, t = obj_, True, pos+t_*dir, t_ # Only update values if the hit object is closer
        if len(hit_lst) == 0 or not hit: # If it hits nothing
            theta, phi = np.arccos(dir.z), np.arctan2(dir.y, dir.x)
            col = settings.sample_bg(theta, phi)
        else: col = obj.color(hit_pt)
    else:
        y0 = np.array([X0[0], X0[1], X0[2], X0[3], V0[0], V0[1], V0[2], V0[3]])
        hit_pt, message, obj = integrator(settings, y0, 1000)
        # Calculate hit point
        hit_pt = GravMethod.mink_pos(np.array([hit_pt[0], hit_pt[1], hit_pt[2], hit_pt[3]]))
        hit_pt = Vec(hit_pt[1], hit_pt[2], hit_pt[3])

        if message == "background": # If light ray escapes, i.e. hits background
            cp = (hit_pt - settings.cam_pos).normal()
            theta, phi = np.arccos(cp.z), np.arctan2(cp.y, cp.x)
            col = settings.sample_bg(theta, phi)
        elif message == "singularity": col = Vec(0,0,0) # If light ray hits a singularity return a black color
        elif message == "hit object": col = obj.color(hit_pt) # If it hits something
        
    return col

@njit(parallel=True, nogil=True, debug=False) # Set debug=True for debugging
def first_render(settings:RenderSettings, pbar:ProgressBar): # First render
    settings.cam_dir.is_normal()
    img = np.zeros((settings.h, settings.w, 3), dtype=np.float32)
    
    for i in prange(settings.w * settings.h):
        x, y = i//settings.h, i%settings.h
        ray_dir = settings.ray_dir_px(x, y)
        col = trace(settings.cam_pos, ray_dir, settings)
        img[y,x] = np.clip(col.np_array(), 0, 1) # Raise to the power of 1/2.2 if human perception of color is to be accounted for
        pbar.update(1)

    return (img*255).astype(np.uint8)

def render_seq(settings:RenderSettings):
    with ProgressBar(total=settings.w*settings.h) as pbar:
        img = first_render(settings, pbar)
    return Image.fromarray(img)
