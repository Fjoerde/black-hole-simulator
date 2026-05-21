# A place for testing out functions/methods that would not otherwise be directly called.
import numpy as np
from PIL import Image
from itertools import product

from Classes.math import *
from Classes.physics import *
from Classes.int_and_settings import *
from Classes.tags import *
from rendering import *
from diagnostics import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

print("hi2")
patch = Patch([np.array([0], dtype=np.float64),
               np.linspace(-30, 10, 11, dtype=np.float64),
               np.linspace(-30, 10, 11, dtype=np.float64),
               np.linspace(-30, 10, 11, dtype=np.float64)])
grid = Grid(patch)

grid = grid.add_patch(Patch([np.array([0], dtype=np.float64),
                             np.linspace(-6, 6, 13, dtype=np.float64),
                             np.linspace(-6, 6, 13, dtype=np.float64),
                             np.linspace(-6, 6, 13, dtype=np.float64)]))

print("hi")
def get_gas(pts):
    x, y, z = pts[:,1], pts[:,2], pts[:,3]
    r = np.sqrt(x**2 + y**2 + z**2)
    temp = 5e4 * np.exp(-r**2/10)
    ext_coeff = 1 * np.exp(-r**2/10)
    gas_params = np.zeros((len(pts), 6), dtype=np.float64); gas_params[:,0] += 1
    gas_params[:,4] = temp; gas_params[:,5] = ext_coeff
    return gas_params

gas_vals = get_gas(grid.pts)
gas = Function(grid, gas_vals)

bg = np.array(Image.open("Images/background1.jpg")).astype(np.float64) / 255.
settings = RenderSettings(w=100, h=100, cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), cam_vel=Vec(0,0,0),
                          background=bg, gas=gas)

print("Calculating geodesics...")
integrator = Integrator(tag=INTEGRATOR_GEODESICEQ, grav_field=settings.grav_field, scene=settings.scene,
                        cam_pos=settings.cam_pos, bg_rad=settings.bg_rad)
geodesics_ = [Function() for _ in range(settings.w * settings.h)]
gas_vals = geodesics_.copy(); spec_ints_ = geodesics_.copy()

with ProgressBar(total=settings.w*settings.h) as pbar:
    geodesics = get_geodesics(integrator, geodesics_, settings, pbar)

print("Finding gas values along geodesics...")
for i in prange(len(geodesics)):
    geodesic = geodesics[i]
    max_t = np.max(geodesic.grid.pts)
    grid = Grid(Patch([np.linspace(0, max_t, 50)]))
    xs = geodesic.interp(grid.pts)[:,:4]
    print(i)
    gas_vals[i] = Function(grid, gas.interp(xs))

print("Calculating colors...")
integrator = Integrator(tag=INTEGRATOR_SPECINT, specint_grid=settings.col_converter.grid,
                        geodesics=geodesics, gas_vals=gas_vals, grav_field=settings.grav_field,
                        obs_vel=settings.grav_field.timelike_cond(settings.cam_vel, settings.cam_pos.four_vec(0)))
with ProgressBar(total=settings.w*settings.h) as pbar:
    spec_ints, cols = get_colors(integrator, settings, spec_ints_, pbar)
# np.save("cols_actual.npy", cols)
render_img = Image.fromarray(cols.reshape(settings.h, settings.w, 3).astype(np.uint8))
render_img.show()