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
disk = [Hittable(tag=HITTABLE_BLACKBODY,
                 shape=Shape(pos=Vec(0,0,0), rot=(np.pi,-np.pi/20,0), tag=SHAPE_ANNULUS, r_in=2.5, r_out=5, height=0.25), temp=1e4)]
kerr = GravField(tag=GRAVFIELD_KERR, pos=Vec(0,0,0), M=0.5, J=0.15)
settings = RenderSettings(w=1280, h=720,
#                          grav_field=kerr, # <-- Un-comment this line
                          cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), cam_vel=Vec(0,0,0), scene=disk,)

t3 = time.perf_counter()
print(f"Initialization finished in {t3-t2:.4f} s\n")

print("Rendering (ignore NumbaPerformanceWarning's)...")
img = render_seq(settings)
filename = "test20.png"
img.save(f"Images/{filename}") 
  
t4 = time.perf_counter()
print(f"Rendering finished in {(t4-t3)/60:.4f} min\n")
print(f"Saved '{filename}'")
