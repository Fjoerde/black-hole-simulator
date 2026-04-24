import numpy as np
from PIL import Image
from numba import njit, prange
from numba_progress import ProgressBar

from render.Classes.base import *
from render.Classes.tags import *
from render.Classes.classes import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

@njit(fastmath=True)
def trace(integrator:Integrator, y0:np.ndarray, bg_rad:float) -> np.ndarray:
    """Returns a null geodesic representing the path of a light ray that ends up at the camera, given over regular intervals.
    Path given in Minkowski coordinates.
    
    y0: The initial state vector used to perform ray-marching.
    
    bg_rad: The radius of the background box."""
    
    if integrator.tag != INTEGRATOR_GEODESICEQ: raise ValueError("Invalid tag for integrator.")
    geodesic = integrator.solve(0, y0, h_init=bg_rad/50, max_t=bg_rad*5)
    length = np.max(geodesic.grid.pts)
    grid = Grid(Patch(np.linspace(0, length, 101)))
    geodesic = geodesic.resample(grid)
    vals = np.empty((len(grid.pts), 8), dtype=np.float64)
    for i in range(len(grid.pts)):
        vals[i,:4] = integrator.grav_field.mink_pos(geodesic.vals[i,:4])
        vals[i,4:] = -integrator.grav_field.mink_vel(geodesic.vals[i,4:8]) * length
    return np.flipud(vals)

@njit(parallel=True, fastmath=True)
def get_geodesics(settings:RenderSettings, pbar:ProgressBar) -> np.ndarray: 
    """Returns the paths of the light ray over each pixel in the image."""

    integrator = Integrator(tag=INTEGRATOR_GEODESICEQ, grav_field=settings.grav_field, scene=settings.scene, cam_pos=settings.cam_pos)
    geodesics = np.empty((settings.w*settings.h, 101, 8))
    x0 = np.array([0, settings.cam_pos.x, settings.cam_pos.y, settings.cam_pos.z])
    X0 = settings.grav_field.coord_pos(x0)

    for i in prange(settings.w * settings.h):
        x, y = i//settings.h, i%settings.h
        ray_dir = settings.ray_dir_px(x, y)
        V0 = settings.grav_field.null_cond(ray_dir, x0)
        y0 = np.concatenate((X0, V0, [0]))
        geodesics[i] = trace(integrator, y0, settings.bg_rad)
        pbar.update(1)
    return geodesics

@njit(parallel=True, fastmath=True)
def get_colors(geodesics:np.ndarray, settings:RenderSettings, pbar:ProgressBar) -> np.ndarray:
    """Returns the spectral intensity values over a given grid over each pixel in the image.
    
    geodesics: Shape (n_px, 101, 8) storing the positions on and tangent vector to the geodesic (hence 8),
    along 101 different points on the geodesic, for each pixel."""

    n_px = settings.w * settings.h
    grid = settings.col_converter.grid
    vels = settings.gas.vel.interp(geodesics[:,:,:4].reshape(n_px*101, 4)).reshape(n_px, 101, 4)
    temps = settings.gas.temp.interp(geodesics[:,:,:4].reshape(n_px*101, 4)).reshape(n_px, 101)
    ext_coeffs = settings.gas.ext_coeff.interp(geodesics[:,:,:4].reshape(n_px*101, 4)).reshape(n_px, 101)
    grid0 = Grid(Patch([np.linspace(0, 1, 101)]))
    x0 = np.array([0, settings.cam_pos.x, settings.cam_pos.y, settings.cam_pos.z])
    obs_vel = settings.grav_field.timelike_cond(settings.cam_vel, x0)
    img = np.empty((n_px, 3), dtype=np.float64)

    for i in prange(n_px):
        spec_int0 = np.zeros(len(grid.pts), dtype=np.float64)
        x0 = geodesics[i,0,:4]; pos = Vec(x0[1], x0[2], x0[3])
        if (pos - settings.cam_pos).length() >= settings.bg_rad: # If light ray hits the background
            theta = np.acos(pos.z / pos.length()); phi = np.atan2(pos.y, pos.x)
            spec_int0 = settings.col_converter.get_spec_int(settings.sample_bg(theta, phi)).vals
        else: # Find which object the light ray hits
            for obj in settings.scene:
                if obj.shape.on_surface(pos):
                    spec_int0 = obj.spec_int.vals
                    break
        gas = Gas(vel=Function(grid0, vels[i]), temp=Function(grid0, temps[i]), ext_coeff=Function(grid0, ext_coeffs[i]))
        integrator = Integrator(tag=INTEGRATOR_SPECINT, specint_grid=grid, geodesic=Function(grid0, geodesics[i]),
                                gas=gas, grav_field=settings.grav_field, obs_vel=obs_vel)
        spec_int = integrator.solve(0, spec_int0, h_init=0.02, max_t=1, tol=1e-3).vals[-1]
        ray_dir = -geodesics[i,-1,4:]; ray_dir = Vec(ray_dir[1], ray_dir[2], ray_dir[3]) # Recover initial ray direction
        spec_int /= settings.aberration(ray_dir) # Relativistic aberration
        img[i] = settings.col_converter.get_rgb(Function(grid, spec_int)) * 255.
        pbar.update(1)

    return img


def render_seq(settings:RenderSettings) -> Image:
    print("Calculating geodesics...")
    with ProgressBar(total=settings.w*settings.h) as pbar:
        geodesics = get_geodesics(settings, pbar)
    print("Calculating colors...")
    with ProgressBar(total=settings.w*settings.h) as pbar:
        img = get_colors(geodesics, settings, pbar)
    img = img.reshape(settings.w, settings.h, 3)
    return Image.fromarray(img.astype(np.uint8))
