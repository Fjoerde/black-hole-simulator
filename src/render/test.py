# A place for testing out functions/methods that would not otherwise be directly called.

import numpy as np
from PIL import Image
from Classes.math import *
from Classes.physics import *
from Classes.int_and_settings import *
from Classes.tags import *
from motion_helper import *
from img_rendering import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def get_gas(pts):
    x, y, z = pts[:,1], pts[:,2], pts[:,3]
    s = np.sqrt(x**2 + y**2) - 5
    d = np.sqrt(s**2 + z**2)
    temp = 2e4 * np.exp(-d**2/5)
    ext_coeff = 0.125 * np.exp(-d**2/5)
    gas_params = np.zeros((len(pts), 6), dtype=np.float64); gas_params[:,0] += 1 # Pretend here the metric is Minkowski for now
    gas_params[:,4] = temp; gas_params[:,5] = ext_coeff
    return gas_params
grid = Grid(Patch([np.array([0], dtype=np.float64),
                   np.linspace(-30, 10, 11, dtype=np.float64),
                   np.linspace(-30, 10, 11, dtype=np.float64),
                   np.linspace(-30, 10, 11, dtype=np.float64)]))
grid = grid.add_patch(Patch([np.array([0], dtype=np.float64),
                             np.linspace(-6, 6, 13, dtype=np.float64),
                             np.linspace(-6, 6, 13, dtype=np.float64),
                             np.linspace(-6, 6, 13, dtype=np.float64)]))

gas = Function(grid, get_gas(grid.pts))
kerr = GravField(tag=GRAVFIELD_KERRNEWMAN, pos=Vec(0,0,0), M=0.5, J=0.225, Q=0.05)

print("Rendering (ignore NumbaPerformanceWarning's)...")
for i in range(6):
    T1 = time.perf_counter()
    theta = i * np.pi/12
    cam_pos = Vec(-12.5*np.sin(theta), 0, -12.5*np.cos(theta))
    settings = RenderSettings(w=800, h=600, cam_pos=cam_pos, cam_vel=Vec(0,0,0), rot=zero_roll(cam_pos),
                              bg_rad=30, grav_field=kerr, gas=gas)
    x0 = settings.cam_pos.four_vec(settings.t)
    X0 = settings.grav_field.coord_pos(x0)
    integrator = Integrator(tag=INTEGRATOR_GEODESICEQ, grav_field=settings.grav_field, scene=settings.scene,
                            cam_pos=settings.cam_pos, bg_rad=settings.bg_rad)
    geodesics = [Function() for _ in range(settings.w * settings.h)]
    for i in range(settings.w * settings.h):
        y, x = i//settings.w, i%settings.w
        ray_dir = settings.ray_dir_px(x, y)
        print("hi")
        V0 = settings.grav_field.null_cond(ray_dir, x0)
        y0 = np.concatenate((X0, V0))
        print(y0)
        geodesics[i] = trace_geodesic(integrator, y0, settings.bg_rad)    

    T2 = time.perf_counter()