import numpy as np
from PIL import Image
from numba import njit, prange
from numba_progress import ProgressBar

from Classes.math import *
from Classes.tags import *
from Classes.physics import *
from Classes.int_and_settings import *
from plots import *

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
    geodesic = integrator.solve(0, y0, bg_rad/50, bg_rad*3)
    gdsic_vals = np.empty((len(geodesic.grid.pts), 9), dtype=np.float64)
    for i in range(len(geodesic.grid.pts)):
        x0 = geodesic.vals[i,:4]
        gdsic_vals[i,:4] = integrator.grav_field.mink_pos(x0)
        gdsic_vals[i,4:8] = -integrator.grav_field.mink_vel(geodesic.vals[i,4:8], x0)
        gdsic_vals[i,8] = geodesic.vals[i,8]
    gdsic_vals = np.ascontiguousarray(np.flipud(gdsic_vals))
    grid_pts = geodesic.grid.pts.reshape(len(geodesic.grid.pts))
    grid_pts = np.ascontiguousarray(np.flip(np.max(grid_pts) - grid_pts))
    geodesic = Function(Grid(Patch([grid_pts])), gdsic_vals)
    return geodesic

@njit(parallel=True, fastmath=True)
def get_geodesics(integrator:Integrator, geodesics:list[Function], settings:RenderSettings, pbar:ProgressBar) -> list[Function]:
    """Returns the paths of the light ray over each pixel in the image."""

    x0 = settings.cam_pos.four_vec(0)
    X0 = settings.grav_field.coord_pos(x0)
    for i in prange(settings.w * settings.h):
        y, x = i//settings.w, i%settings.w
        ray_dir = settings.ray_dir_px(x, y)
        V0 = settings.grav_field.null_cond(ray_dir, x0)
        y0 = np.concatenate((X0, V0, np.array([0], dtype=np.float64)))
        geodesics[i] = trace_geodesic(integrator, y0, settings.bg_rad)
        pbar.update(1)
    return geodesics

@njit(parallel=True, fastmath=True)
def get_gas_vals(geodesics:list[Function], gas:Function, gas_vals:list[Function], n:int, pbar:ProgressBar) -> list[Function]:
    """Returns the list of gas values along the geodesic."""

    for i in prange(len(geodesics)):
        geodesic = geodesics[i]
        max_t = np.max(geodesic.grid.pts)
        grid = Grid(Patch([np.linspace(0, max_t, n)]))
        xs = geodesic.interp(grid.pts)[:,:4]
        gas_vals[i] = Function(grid, gas.interp(xs))
        pbar.update(1)
    return gas_vals

@njit(parallel=True, fastmath=True)
def get_colors(geodesics:list[Function], gas_vals:list[Function], settings:RenderSettings,
               spec_ints:list[Function], pbar:ProgressBar) -> tuple[Function, np.ndarray]:
    """Returns the spectral intensity values over a given grid for every given geodesic.
    
    geodesics: A list of functions that give the four-positions, tangent vectors (in the future direction), and
    extinction coefficients along the geodesic."""

    n = settings.w * settings.h
    cols = np.zeros((n, 3), dtype=np.float64)
    for i in prange(n):
        integrator = Integrator(INTEGRATOR_SPECINT, settings.grav_field, settings.scene, settings.cam_pos, settings.bg_rad,
                                settings.col_converter.grid, geodesics[i], gas_vals[i],
                                settings.grav_field.timelike_cond(settings.cam_vel, settings.cam_pos.four_vec(0)))
        spec_int0 = np.zeros(len(integrator.specint_grid.pts), dtype=np.float64)
        geodesic = geodesics[i]
        x0 = geodesic.vals[0,:4]; pos = Vec(x0[1], x0[2], x0[3])
        k0 = geodesic.vals[0,4:8]; k1 = geodesic.vals[-1,4:8]
        # Find which object the light ray hits
        if (pos - settings.cam_pos).length() >= settings.bg_rad: # If light ray hits the background
            theta = np.acos(pos.z / pos.length()); phi = np.atan2(pos.y, pos.x)
            spec_int0 = settings.sample_bg(theta, phi)
            spec_int0 = settings.doppler_spec(spec_int0, x0, k0, k1).vals
            spec_int0 = spec_int0.reshape(len(spec_int0))
        else: # Find which object the light ray hits
            for obj in settings.scene:
                if obj.shape.on_surface(pos):
                    spec_int0 = obj.spec_int(pos)
                    spec_int0 = settings.doppler_spec(spec_int0, x0, k0, k1).vals
                    spec_int0 = spec_int0.reshape(len(spec_int0))
                    break
        max_t = np.max(geodesic.grid.pts)
        spec_int = integrator.solve(0, spec_int0, max_t/50, max_t, 10)
        spec_int = spec_int.vals[-1]; spec_int = spec_int.reshape(len(spec_int), 1)
        spec_int = settings.rel_aberr(Function(integrator.specint_grid, spec_int), k1)
        spec_ints[i] = spec_int; cols[i] = settings.col_converter.get_rgb(spec_int) * 255.
        pbar.update(1)
    return spec_ints, cols


# Functions outside of nopython mode
def render_seq(settings:RenderSettings) -> tuple[Image.Image]:
    """Call to perform the rendering sequence."""

    print("Calculating geodesics...")
    gdsic_int = Integrator(tag=INTEGRATOR_GEODESICEQ, grav_field=settings.grav_field, scene=settings.scene,
                            cam_pos=settings.cam_pos, bg_rad=settings.bg_rad)
    geodesics_ = [Function() for _ in range(settings.w * settings.h)]
    gas_vals_ = geodesics_.copy(); spec_ints_ = geodesics_.copy()
    with ProgressBar(total=settings.w*settings.h) as pbar:
        geodesics = get_geodesics(gdsic_int, geodesics_, settings, pbar)
    del gdsic_int

    print("Plotting angular deviations...")
    ang_dev_img = deviation(geodesics, settings)

    print("Finding gas values along geodesics...")
    with ProgressBar(total=settings.w*settings.h) as pbar:
        gas_vals = get_gas_vals(geodesics, settings.gas, gas_vals_, 50, pbar)

    print("Calculating colors...")
    with ProgressBar(total=settings.w*settings.h) as pbar:
        spec_ints, cols = get_colors(geodesics, gas_vals, settings, spec_ints_, pbar)
    render_img = Image.fromarray(cols.reshape(settings.h, settings.w, 3).astype(np.uint8))
    render_img.show()
    del geodesics, gas_vals

    print("Plotting the average spectral radiance...")
    avg_specint_img = avg_specint(spec_ints, settings)

    return render_img, ang_dev_img, avg_specint_img
