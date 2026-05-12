import numpy as np
from PIL import Image
from numba import njit, prange
from numba_progress import ProgressBar

from Classes.math import *
from Classes.tags import *
from Classes.physics import *
from Classes.int_and_settings import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Njitted functions
@njit(fastmath=True)
def trace_geodesic(integrator:Integrator, y0:np.ndarray, bg_rad:float) -> Function:
    """Returns a null geodesic representing the path of a light ray that ends up at the camera, given over regular intervals.
    Path is given in Minkowski coordinates.
    
    y0: The initial state vector used to perform ray-marching.
    
    bg_rad: The radius of the background box."""
    
    if integrator.tag != INTEGRATOR_GEODESICEQ: raise ValueError("Invalid tag for integrator.")
    geodesic, gas_param = integrator.solve(0, y0, bg_rad/50, bg_rad*3)
    gdsic_vals = np.empty((len(geodesic.grid.pts), 8), dtype=np.float64)
    for i in range(len(geodesic.grid.pts)):
        x0 = geodesic.vals[i,:4]
        gdsic_vals[i,:4] = integrator.grav_field.mink_pos(x0)
        gdsic_vals[i,4:] = -integrator.grav_field.mink_vel(geodesic.vals[i,4:], x0)
    gdsic_vals = np.ascontiguousarray(np.flipud(gdsic_vals))
    gas_vals = np.ascontiguousarray(np.flipud(gas_param.vals))
    new_vals = np.hstack((gdsic_vals, gas_vals))

    grid_pts = geodesic.grid.pts.reshape(len(geodesic.grid.pts))
    grid_pts = np.ascontiguousarray(np.flip(np.max(grid_pts) - grid_pts))
    geodesic = Function(Grid(Patch([grid_pts])), new_vals)
    return geodesic

@njit(parallel=True, fastmath=True)
def get_geodesics(integrator:Integrator, geodesics:list[Function], settings:RenderSettings, pbar:ProgressBar) -> list[Function]:
    """Returns the paths of the light ray over each pixel in the image."""

    x0 = np.ascontiguousarray(np.array([0, settings.cam_pos.x, settings.cam_pos.y, settings.cam_pos.z]))
    X0 = settings.grav_field.coord_pos(x0)
    for i in prange(settings.w * settings.h):
        y, x = i//settings.w, i%settings.w
        ray_dir = settings.ray_dir_px(x, y)
        V0 = settings.grav_field.null_cond(ray_dir, x0)
        y0 = np.concatenate((X0, V0))
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
    cols = np.zeros((n, 3), dtype=np.float64)
    for i in prange(n):
        spec_int0 = np.zeros(len(integrator.specint_grid.pts), dtype=np.float64)
        geodesic = integrator.geodesics[i]
        x0 = geodesic.vals[0,:4]; pos = Vec(x0[1], x0[2], x0[3])
        # Find which object the light ray hits
        for obj in settings.scene:
            if obj.shape.on_surface(pos):
                spec_int0 = obj.spec_int(pos).vals
                spec_int0 = spec_int0.reshape(len(spec_int0))
                break
        integrator.solve_idx = i; max_t = np.max(geodesic.grid.pts)
        spec_int, _ = integrator.solve(0, spec_int0, max_t/50, max_t, 1e-4)
        spec_int = spec_int.vals[-1]; spec_int = spec_int.reshape(len(spec_int), 1)
        spec_int = Function(integrator.specint_grid, spec_int)
        cols[i] = settings.col_converter.get_rgb(spec_int) * 255.
        pbar.update(1)
    return cols


# Function to actually call
def render_seq(settings:RenderSettings) -> Image:
    """Call to perform the rendering sequence."""

    print("Calculating geodesics...")
    integrator = Integrator(tag=INTEGRATOR_GEODESICEQ, grav_field=settings.grav_field, scene=settings.scene,
                            cam_pos=settings.cam_pos, bg_rad=settings.bg_rad, gas=settings.gas)
    geodesics = [Function() for _ in range(settings.w * settings.h)]
    with ProgressBar(total=settings.w*settings.h) as pbar:
        geodesics = get_geodesics(integrator, geodesics, settings, pbar)
    print("Calculating colors...")
    x0 = np.array([0, settings.cam_pos.x, settings.cam_pos.y, settings.cam_pos.z])
    integrator = Integrator(tag=INTEGRATOR_SPECINT, specint_grid=settings.col_converter.grid,
                            geodesics=geodesics, grav_field=settings.grav_field,
                            obs_vel=settings.grav_field.timelike_cond(settings.cam_vel, x0))
    with ProgressBar(total=settings.w*settings.h) as pbar:
        cols = get_colors(integrator, settings, pbar)
    img = Image.fromarray(cols.reshape(settings.h, settings.w, 3).astype(np.uint8))
    img.show()
    return img
