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

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
t2 = time.perf_counter()
print(f"Lowering finished in {t2-t1:.4f} s\n")

print("Initializing...")
bg = np.array(Image.open("Images/background1.jpg")).astype(np.float64) / 255.
settings = RenderSettings(w=1280, h=720, cam_pos=Vec(0,0,0), cam_vel=Vec(0,0,0), rot=(0, np.pi/2, 0), background=bg)

t3 = time.perf_counter()
print(f"Initialization finished in {t3-t2:.4f} s\n")

print("Rendering (ignore NumbaPerformanceWarning's)...")
for i in range(-2, 3):
    settings.cam_vel = Vec(0.25*i, 0, 0)
    rendered_img, _, _ = render_img(settings)
    rendered_img.save(f"Images/aberration{i}.png")

t4 = time.perf_counter()
print(f"Rendering finished in {(t4-t3)/60:.4f} min\n")

