# A place for testing out functions/methods that would not otherwise be directly called.

import numpy as np
from PIL import Image
from Classes.math import *
from Classes.physics import *
from Classes.int_and_settings import *
from Classes.tags import *
from motion_helper import *
from vid_rendering import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def get_gas(pts):
    x, y, z = pts[:,1], pts[:,2], pts[:,3]
    s = np.sqrt(x**2 + y**2) - 5
    d = np.sqrt(s**2 + z**2)
    temp = 2e4 * np.exp(-d**2/3)
    ext_coeff = 0.125 * np.exp(-d**2/3)
    gas_params = np.zeros((len(pts), 6), dtype=np.float64); gas_params[:,0] += 1
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
bg = np.array(Image.open("Images/background1.jpg")).astype(np.float64) / 255.
ss = GravField(tag=GRAVFIELD_SCHWARZSCHILD, pos=Vec(0,0,0), M=0.5)
cam_worldline = get_obj_path(ss, Vec(-10,0,0), Vec(0,0.25,0.25)/np.sqrt(2), 0, 400)
tau_scale = np.max(cam_worldline.grid.pts) / 5
rots1 = look_to_origin(cam_worldline, tau_scale, 1, 5)
# rots2 = look_to_origin(cam_worldline, tau_scale, 24, 120)
settings = VidSettings(w=100, h=100, cam_worldline=cam_worldline, tau_scale=tau_scale, frame_num=5, rots=rots1, fps=1, background=bg, grav_field=ss)
frame_num = 0; geodesics = [Function() for _ in range(settings.w * settings.h)]
integrator = Integrator(tag=INTEGRATOR_GEODESICEQ, grav_field=settings.grav_field, scene=settings.scene,
                        cam_pos=settings.cam_pos[0], bg_rad=settings.bg_rad)
print("hi")
x0 = settings.cam_pos[frame_num].four_vec(settings.t[frame_num])
print(x0)
print("hi2")
X0 = settings.grav_field.coord_pos(x0)
print("hi3")
for i in range(settings.w * settings.h):
    y, x = i//settings.w, i%settings.w
    ray_dir = settings.ray_dir_px(x, y, frame_num)
    V0 = settings.grav_field.null_cond(ray_dir, x0)
    y0 = np.concatenate((X0, V0))
    geodesics[i] = trace_geodesic(integrator, y0, settings.bg_rad)
