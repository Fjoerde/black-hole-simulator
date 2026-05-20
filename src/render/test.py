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
np.set_printoptions(threshold=np.inf)
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
settings = RenderSettings(w=100, h=100, cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), cam_vel=Vec(0.1,0,0),
                          background=bg, gas=gas)

print("Calculating geodesics...")
integrator = Integrator(tag=INTEGRATOR_GEODESICEQ, grav_field=settings.grav_field, scene=settings.scene,
                        cam_pos=settings.cam_pos, bg_rad=settings.bg_rad)
geodesics = [Function() for _ in range(settings.w * settings.h)]
with ProgressBar(total=settings.w*settings.h) as pbar:
    geodesics = get_geodesics(integrator, geodesics, settings, pbar)

print("Finding gas values along geodesics...")
gas_vals = geodesics.copy()
with ProgressBar(total=settings.w*settings.h) as pbar:
    gas_vals = get_gas_vals(geodesics, settings.gas, gas_vals, 50, pbar)

print("Calculating colors...")
x0 = np.array([0, settings.cam_pos.x, settings.cam_pos.y, settings.cam_pos.z])
integrator = Integrator(tag=INTEGRATOR_SPECINT, specint_grid=settings.col_converter.grid,
                        geodesics=geodesics, gas_vals=gas_vals, grav_field=settings.grav_field,
                        obs_vel=settings.grav_field.timelike_cond(settings.cam_vel, x0))
spec_ints = geodesics.copy()
n = len(integrator.geodesics)
cols = np.zeros((n, 3), dtype=np.float64)
for i in range(n):
    spec_int0 = np.zeros(len(integrator.specint_grid.pts), dtype=np.float64)
    geodesic = integrator.geodesics[i]
    x0 = geodesic.vals[0,:4]; pos = Vec(x0[1], x0[2], x0[3])
    k0 = geodesic.vals[0,4:8]; k1 = geodesic.vals[-1,4:8]
    # Find which object the light ray hits
    if (pos - settings.cam_pos).length() >= settings.bg_rad: # If light ray hits the background
        theta = np.acos(pos.z / pos.length()); phi = np.atan2(pos.y, pos.x)
        spec_int0 = settings.sample_bg(theta, phi)
        spec_int0 = settings.doppler_spec(spec_int0, x0, k0, k1).vals.flatten()
    else: # Find which object the light ray hits
        for obj in settings.scene:
            if obj.shape.on_surface(pos):
                spec_int0 = obj.spec_int(pos)
                spec_int0 = settings.doppler_spec(spec_int0, x0, k0, k1).vals.flatten()
                break
    integrator.solve_idx = i; max_t = np.max(geodesic.grid.pts)
    spec_int = integrator.solve(0, spec_int0, max_t/50, max_t, 1e-4)
    spec_int = spec_int.vals[-1]; spec_int = spec_int.reshape(len(spec_int), 1)
    spec_int = settings.rel_aberr(Function(integrator.specint_grid, spec_int), k1)
    spec_ints[i] = spec_int; cols[i] = settings.col_converter.get_rgb(spec_int) * 255.
    print(i)

render_img = Image.fromarray(cols.reshape(settings.h, settings.w, 3).astype(np.uint8))
render_img.show()