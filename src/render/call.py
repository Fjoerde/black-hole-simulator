#### READ INSTRUCTIONS IN COMMENTS. THERE ARE A TOTAL OF TWO RUNS TO BE DONE. ####
import numpy as np
from PIL import Image
import time
from numba import config, set_num_threads
set_num_threads(config.NUMBA_DEFAULT_NUM_THREADS)

print("Importing and lowering code...")
t1 = time.perf_counter()
from Classes.math import *
from Classes.physics import GravField
from Classes.int_and_settings import *
from Classes.tags import *
from motion_helper import *
from img_rendering import render_img
# from vid_rendering import render_vid

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
t2 = time.perf_counter()
print(f"Lowering finished in {t2-t1:.4f} s\n")

print("Initializing...")
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

t3 = time.perf_counter()
print(f"Initialization finished in {t3-t2:.4f} s\n")

print("Rendering (ignore NumbaPerformanceWarning's)...")
for i in range(6):
    T1 = time.perf_counter()
    theta = i * np.pi/12 + 1e-2
    cam_pos = Vec(-12.5*np.sin(theta), 0, -12.5*np.cos(theta))
    settings = RenderSettings(w=800, h=600, cam_pos=cam_pos, cam_vel=Vec(0,0,0), rot=zero_roll(cam_pos),
                              bg_rad=30, grav_field=kerr, gas=gas)
    rendered_img, ang_dev_img, tot_specint_img = render_img(settings)
    rendered_img.save(f"Images/render_kerr{i}")
    ang_dev_img.save(f"Images/ang_dev_kerr{i}")
    tot_specint_img.save(f"Images/tot_specint_kerr{i}")
    T2 = time.perf_counter()
    print(f"Rendered image with theta = {theta} in {(T2-T1)/60:.4f} min")

t4 = time.perf_counter()
print(f"Rendering finished in {(t4-t3)/60:.4f} min\n")

