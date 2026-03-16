import numpy as np
from PIL import Image
from numba import njit, prange
from numba_progress import ProgressBar

from render.Classes.base import *
from render.Classes.tags import *
import render.Classes.GravField.methods as GravMethod
from render.Classes.classes import RenderSettings

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

@njit
def rk4_step(t, y, h, settings:RenderSettings): # h is a step size
    k1 = GravMethod.geodesic_eq(t, y, settings.grid, settings.Gamma_grid, settings.neighbors)
    k2 = GravMethod.geodesic_eq(t+h/2, y+h*k1/2, settings.grid, settings.Gamma_grid, settings.neighbors)
    k3 = GravMethod.geodesic_eq(t+h/2, y+h*k2/2, settings.grid, settings.Gamma_grid, settings.neighbors)
    k4 = GravMethod.geodesic_eq(t+h, y+h*k3, settings.grid, settings.Gamma_grid, settings.neighbors)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

@njit
def integrator(settings:RenderSettings, y0:float, max_t:float=1000,
               tol:float=1e-8, h_init:float=0.1, safety:float=0.9, threshold:float=1e4):
    """Returns the point and the object the light ray hits."""
    t, y, h = 0., y0, h_init
    hit_obj, message = settings.scene[0], "background"

    while t < max_t:

        # Check for any hits
        dists = np.empty(len(settings.scene), dtype=np.float64)
        for i in range(len(settings.scene)): # If the distance of closest approach is < 1e-4 or already inside
            obj = settings.scene[i]
            dist = obj.shape.in_shape_int(y)
            if np.abs(dist) < 1e-4: return y, "hit object", obj
            dists[i] = dist
        if settings.escape(y) > 0: return y, "background", settings.scene[0]
        if GravMethod.singularity(y, settings.grid, settings.g_grid, settings.neighbors) > threshold:
            return y, "singularity", settings.scene[0]

        # Adapt step size to how close the point is to any object
        speed = np.linalg.norm(GravMethod.geodesic_eq(t, y, settings.grid, settings.Gamma_grid, settings.neighbors))
        min_dist = min(dists)
        h = min(h, 2*min_dist/speed)

        # Perform integration
        if t + h > max_t: h = max_t - t # Prevents overstepping
        y_coarse = rk4_step(t, y, h, settings) # Coarse step
        y_mid = rk4_step(t, y, h/2, settings)
        y_fine = rk4_step(t+h/2, y_mid, h/2, settings) # Fine step made of 2 substeps
        # Error estimate
        error = np.max(np.abs(y_coarse - y_fine)) / 15
        if error <= tol:
            t += h
            y = y_fine
            if error < 1e-15 : h *= safety * 5.
            else:
                factor = (tol / error) ** (1/5)
                if factor < 0.1: h *= safety * 0.1
                elif 0.1 < factor < 5.0: h *= safety * factor
                else: h *= safety * 5.
        else: # Reject solution and reduce step size
            factor = (tol / error) ** (1/5)
            if factor > 0.1: h *= safety * factor
            else: h *= safety * 0.1

    return y, message, hit_obj

@njit
def trace(pos:Vec, dir:Vec, settings:RenderSettings): # Returns color the light ray starting from pos in dir 'sees'
    dir.is_normal()
    col = Vec(0,0,0)
    X0 = GravMethod.coord_pos(np.array([0, pos.x, pos.y, pos.z]))
    V0 = GravMethod.normed(dir, np.array([0, pos.x, pos.y, pos.z]))
    y0 = np.array([X0[0], X0[1], X0[2], X0[3], V0[0], V0[1], V0[2], V0[3]])
    hit_pt, message, obj = integrator(settings, y0, max_t=1000, threshold=1e3)

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
