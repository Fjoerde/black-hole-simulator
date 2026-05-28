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
from vid_rendering import render_vid

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
vid_length = 5 # in seconds
tau_scale = vid_length / np.max(cam_worldline.grid.pts)
rots = look_to_origin(cam_worldline, tau_scale, 24, 120)
settings = VidSettings(w=800, h=600, cam_worldline=cam_worldline, tau_scale=tau_scale, frame_num=120, rots=rots, fps=24, background=bg, grav_field=ss, gas=gas)

t3 = time.perf_counter()
print(f"Initialization finished in {t3-t2:.4f} s\n")

print("Rendering (ignore NumbaPerformanceWarning's)...")
render_vid(settings, save_frames="Video Frames")

t4 = time.perf_counter()
print(f"Rendering finished in {(t4-t3)/60:.4f} min\n")

