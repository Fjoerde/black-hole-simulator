import numpy as np
from PIL import Image
import time

print("Importing and lowering code...")
t1 = time.perf_counter()
from render.Classes.math import *
from render.Classes.physics import Shape, Hittable, GravField
from render.Classes.int_and_settings import RenderSettings
from render.Classes.tags import *
from render.rendering import render_seq

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
t2 = time.perf_counter()
print(f"Lowering finished in {t2-t1:.4f} s\n")

print("Initializing...")
scene = [Hittable(tag=HITTABLE_BLACKBODY,
                  shape=Shape(pos=Vec(10,0,0), rot=(0,0,0), tag=SHAPE_SPHERE, radius=3), temp=3e3)]
disk = [Hittable(tag=HITTABLE_BLACKBODY,
                 shape=Shape(pos=Vec(10,0,0), rot=(np.pi,-np.pi/20,0), tag=SHAPE_ANNULUS, r_in=2.5, r_out=5), temp=3e3)]
# bg = np.array(Image.open("Images/background2.jpg")).astype(np.float64) / 255.0
settings = RenderSettings(w=1280, h=720,
                          cam_pos=Vec(0,0,0), cam_dir=Vec(1,0,0), cam_vel=Vec(0,0,0),
                          scene=disk, bg_rad=20)

t3 = time.perf_counter()
print(f"Initialization finished in {t3-t2:.4f} s\n")

print("Rendering...")
img = render_seq(settings)
filename = "test20.png"
img.save(f"Images/{filename}") 
  
t4 = time.perf_counter()
print(f"Rendering finished in {(t4-t3)/60:.4f} min\n")
print(f"Saved '{filename}'")
