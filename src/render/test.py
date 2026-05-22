# A place for testing out functions/methods that would not otherwise be directly called.
import numpy as np
from PIL import Image
from Classes.physics import *
from Classes.int_and_settings import RenderSettings
from Classes.tags import *
from diagnostics import *

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

bg = np.array(Image.open("Images/background1.jpg")).astype(np.float64) / 255.
ss = GravField(tag=GRAVFIELD_SCHWARZSCHILD, pos=Vec(0,0,0), M=0.5)
settings = RenderSettings(w=100, h=100, cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), cam_vel=Vec(0.5,0,0), background=bg, grav_field=ss)

print(settings.ray_dir_px(50, 80).np_array())
print(settings.ray_dir_px(20, 50).np_array())
geodesic1 = look_ray(settings.cam_pos, settings.ray_dir_px(50, 80), settings)
geodesic2 = look_ray(settings.cam_pos, settings.ray_dir_px(20, 50), settings)
print(geodesic1.vals[0,-1])
print(geodesic2.vals[0,-1])