import numpy as np
from PIL import Image
from numba import njit, prange
from numba_progress import ProgressBar

from render.Classes.math import *
from render.Classes.tags import *
from render.Classes.physics import *
from render.Classes.int_and_settings import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

@njit(fastmath=True)
def trace_geodesic(integrator:Integrator, y0:np.ndarray, bg_rad:float) -> np.ndarray:
    """Returns a null geodesic representing the path of a light ray that ends up at the camera, given over regular intervals.
    Path given in Minkowski coordinates.
    
    y0: The initial state vector used to perform ray-marching.
    
    bg_rad: The radius of the background box."""
    
    if integrator.tag != INTEGRATOR_GEODESICEQ: raise ValueError("Invalid tag for integrator.")
    geodesic = integrator.solve(0, y0, bg_rad/50, bg_rad*3)
    vals = np.empty((len(geodesic.grid.pts), 9), dtype=np.float64)
    for i in range(len(geodesic.grid.pts)):
        x0 = geodesic.vals[i,:4]
        vals[i,:4] = integrator.grav_field.mink_pos(x0)
        vals[i,4:8] = -integrator.grav_field.mink_vel(geodesic.vals[i,4:8], x0)
        vals[i,8] = geodesic.vals[i,8]
    vals = np.ascontiguousarray(np.flipud(vals))
    grid_pts = geodesic.grid.pts.reshape(len(geodesic.grid.pts))
    grid_pts = np.ascontiguousarray(np.flip(np.max(grid_pts) - grid_pts))
    func = Function(Grid(Patch([grid_pts])), vals)
    return func

@njit(parallel=True, fastmath=True)
def get_geodesics(integrator:Integrator, geodesics:list[Function], settings:RenderSettings, pbar:ProgressBar) -> np.ndarray: 
    """Returns the paths of the light ray over each pixel in the image."""

    x0 = np.ascontiguousarray(np.array([0, settings.cam_pos.x, settings.cam_pos.y, settings.cam_pos.z]))
    X0 = settings.grav_field.coord_pos(x0)
    for i in prange(settings.w * settings.h):
        x, y = i//settings.h, i%settings.h
        ray_dir = settings.ray_dir_px(x, y)
        V0 = settings.grav_field.null_cond(ray_dir, x0)
        y0 = np.concatenate((X0, V0, np.array([0], dtype=np.float64)))
        geodesics[i] = trace_geodesic(integrator, y0, settings.bg_rad)
        pbar.update(1)
    return geodesics    

@njit(parallel=True, fastmath=True)
def get_colors(integrator:Integrator, settings:RenderSettings, pbar:ProgressBar) -> np.ndarray:
    """Returns the spectral intensity values over a given grid for every given geodesic.
    
    geodesics: A list of functions that give the four-positions, tangent vectors (in the future direction), and
    extinction coefficients along the geodesic."""

    if integrator.tag != INTEGRATOR_SPECINT: raise ValueError("Invalid tag for integrator.")

    n = len(integrator.geodesics)
    cols = np.empty((n, 3), dtype=np.float64)
    for i in prange(n):
        spec_int0 = np.zeros(len(integrator.specint_grid.pts), dtype=np.float64)
        geodesic = integrator.geodesics[i]
        x0 = geodesic.vals[0,:4]; pos = Vec(x0[1], x0[2], x0[3])
        if (pos - settings.cam_pos).length() >= settings.bg_rad: # If light ray hits the background
            theta = np.acos(pos.z / pos.length()); phi = np.atan2(pos.y, pos.x)
            spec_int0 = settings.col_converter.get_spec_int(settings.sample_bg(theta, phi)).vals
            spec_int0 = spec_int0.reshape(len(spec_int0))
        else: # Find which object the light ray hits
            for obj in settings.scene:
                if obj.shape.on_surface(pos):
                    spec_int0 = obj.spec_int(pos).vals
                    spec_int0 = spec_int0.reshape(len(spec_int0))
                    break
        integrator.solve_idx = i; max_t = np.max(geodesic.grid.pts)
        spec_int = integrator.solve(0, spec_int0, max_t/50, max_t, 1e-4).vals[-1]
        spec_int = Function(integrator.specint_grid, spec_int.reshape(len(spec_int), 1))
        cols[i] = settings.col_converter.get_rgb(spec_int) * 255.
        pbar.update(1)
    return cols

def render_seq(settings:RenderSettings) -> Image:
    print("Calculating geodesics...")
    integrator = Integrator(tag=INTEGRATOR_GEODESICEQ, grav_field=settings.grav_field, scene=settings.scene,
                            cam_pos=settings.cam_pos, bg_rad=settings.bg_rad, gas=settings.gas)
    geodesics = [Function() for _ in range(settings.w * settings.h)]
    with ProgressBar(total=settings.w*settings.h) as pbar:
        geodesics = get_geodesics(integrator, geodesics, settings, pbar)
    print("Calculating colors...")
    x0 = np.array([0, settings.cam_pos.x, settings.cam_pos.y, settings.cam_pos.z])
    integrator = Integrator(tag=INTEGRATOR_SPECINT, specint_grid=settings.col_converter.grid, geodesics=geodesics,
                            gas=settings.gas, grav_field=settings.grav_field,
                            obs_vel=settings.grav_field.timelike_cond(settings.cam_vel, x0))
    with ProgressBar(total=settings.w*settings.h) as pbar:
        img = get_colors(integrator, settings, pbar)
    img = img.reshape(settings.h, settings.w, 3)
    return Image.fromarray(img.astype(np.uint8))
