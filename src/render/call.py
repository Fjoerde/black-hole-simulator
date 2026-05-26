#### READ INSTRUCTIONS IN COMMENTS. THERE ARE A TOTAL OF TWO RUNS TO BE DONE. ####
import numpy as np
from PIL import Image
import time

print("Importing and lowering code...")
t1 = time.perf_counter()
from Classes.math import *
from Classes.physics import GravField
from Classes.int_and_settings import RenderSettings
from Classes.tags import *
from rendering import render_img

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
t2 = time.perf_counter()
print(f"Lowering finished in {t2-t1:.4f} s\n")

print("Initializing...")

def get_gas(pts):
    x, y, z = pts[:,1], pts[:,2], pts[:,3]
    r = np.sqrt(x**2 + y**2 + z**2)
    temp = 2e4 * np.exp(-r**2/20)
    ext_coeff = 0.125 * np.exp(-r**2/20)
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
gas_vals = get_gas(grid.pts)
gas = Function(grid, gas_vals)

bg = np.array(Image.open("Images/background1.jpg")).astype(np.float64) / 255.
kerr = GravField(tag=GRAVFIELD_KERR, pos=Vec(0,0,0), M=0.5, J=0.2, ax=Vec(0,0,1))
col_conv = ColConverter(Grid(Patch([np.linspace(0, 1000, 101, dtype=np.float64)])))
settings = RenderSettings(w=1280, h=720, cam_pos=Vec(-5,0,0), cam_vel=Vec(0,0,0), bg_rad=30, col_converter=col_conv, gas=gas, grav_field=kerr)

t3 = time.perf_counter()
print(f"Initialization finished in {t3-t2:.4f} s\n")

print("Rendering (ignore NumbaPerformanceWarning's)...")
rendered_img, ang_dev_img, specint_img = render_img(settings)
rendered_filename = "test22.png"
angdev_filename = "ang_dev1.png"
specint_filename = "spec_int1.png"
render_img.save(f"Images/{render_filename}") 
ang_dev_img.save(f"Images/{angdev_filename}")
specint_img.save(f"Images/{specint_filename}")
  
t4 = time.perf_counter()
print(f"Rendering finished in {(t4-t3)/60:.4f} min\n")
print(f"Saved '{rendered_filename}' and '{angdev_filename}'")

