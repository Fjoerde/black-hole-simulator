#### READ INSTRUCTIONS IN COMMENTS. THERE ARE A TOTAL OF TWO RUNS TO BE DONE. ####

import numpy as np
import time

print("Importing and lowering code...")
t1 = time.perf_counter()
from Classes.math import *
from Classes.physics import Shape, Hittable, GravField
from Classes.int_and_settings import RenderSettings
from Classes.tags import *
from rendering import render_seq

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
t2 = time.perf_counter()
print(f"Lowering finished in {t2-t1:.4f} s\n")

print("Initializing...")

def get_gas(pts):
    x, y, z = pts[:,1], pts[:,2], pts[:,3]
    r = np.sqrt(x**2 + y**2 + z**2)
    temp = 1e4 * np.exp(-r**2)
    ext_coeff = 10 * np.exp(-r**2)
    gas_params = np.zeros((len(pts), 6), dtype=np.float64); gas_params[:,0] += 1
    gas_params[:,4] = temp; gas_params[:,5] = ext_coeff
    return gas_params
grid = Grid(Patch([np.array([0], dtype=np.float64),
                   np.linspace(-5, 5, 11, dtype=np.float64),
                   np.linspace(-5, 5, 11, dtype=np.float64),
                   np.linspace(-5, 5, 11, dtype=np.float64)]))
grid = grid.add_patch(Patch([np.array([0], dtype=np.float64),
                             np.linspace(-3, 3, 13, dtype=np.float64),
                             np.linspace(-3, 3, 13, dtype=np.float64),
                             np.linspace(-3, 3, 13, dtype=np.float64)]))
gas_vals = get_gas(grid.pts)
gas = Function(grid, gas_vals)

sphere = [Hittable(tag=HITTABLE_CHECKERBOARD,
                 shape=Shape(pos=Vec(0,0,0), rot=(0,0,0), tag=SHAPE_SPHERE, radius=3),
                 col1=np.array([0,0,1]), col2=np.array([0,1,0]))]
disk = [Hittable(tag=HITTABLE_BLACKBODY,
                 shape=Shape(pos=Vec(0,0,0), rot=(np.pi/2,-np.pi/20,0), tag=SHAPE_ANNULUS, r_in=2.5, r_out=5, height=0.25), temp=4000)]
ss = GravField(tag=GRAVFIELD_SCHWARZSCHILD, pos=Vec(0,0,0), M=0.5)
settings1 = RenderSettings(w=1280, h=720, cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), cam_vel=Vec(0,0,0), gas=gas)
settings2 = RenderSettings(w=1280, h=720, cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), cam_vel=Vec(0,0,0), scene=disk, grav_field=ss)

t3 = time.perf_counter()
print(f"Initialization finished in {t3-t2:.4f} s\n")

print("Rendering (ignore NumbaPerformanceWarning's)...")
render_img, ang_dev_img = render_seq(settings1) # <-- Un-comment the next line and comment this line for run 2
# render_img, ang_dev_img = render_seq(settings2) # <-- Un-comment the previous line and comment this line for run 1
render_filename = "test21.png" # <-- test21.png for run 1, test22.png for run 2
ang_dev_filename = "ang_dev1.png" # <-- ang_dev1.png for run 1, ang_dev2.png for run 2
render_img.save(f"Images/{render_filename}") 
ang_dev_img.save(f"Images/{ang_dev_filename}")
  
t4 = time.perf_counter()
print(f"Rendering finished in {(t4-t3)/60:.4f} min\n")
print(f"Saved '{render_filename}' and '{ang_dev_filename}'")
