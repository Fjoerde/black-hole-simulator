# A place for testing out functions/methods that would not otherwise be directly called.
import numpy as np
from PIL import Image
from render.Classes.math import *
from render.Classes.physics import *
from render.Classes.int_and_settings import *
from render.Classes.tags import *
from render.diagnostics import *
from render.rendering import *
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))


bg = np.array(Image.open("Images/background2.jpg")) / 255.0
print("hi")
settings = RenderSettings(w=1280, h=720,
                          cam_pos=Vec(0,0,0), cam_dir=Vec(1,0,0), cam_vel=Vec(0,0,0),
                          bg_rad=20)
print("hi2")
geodesic, gas_param = look_ray(Vec(0,0,0), Vec(1,0,0), settings)
print(geodesic.grid.pts)
print(geodesic.vals)
print(gas_param.vals)

spec_int, col = ray_col(geodesic, gas_param, Vec(0,0,0), settings)
display_specint(spec_int)
display_col(col)


"""
integrator = Integrator(tag=INTEGRATOR_GEODESICEQ, grav_field=settings.grav_field, scene=settings.scene,
                        cam_pos=settings.cam_pos, bg_rad=settings.bg_rad, gas=settings.gas)
geodesics = []
x0 = np.array([0, settings.cam_pos.x, settings.cam_pos.y, settings.cam_pos.z])
X0 = settings.grav_field.coord_pos(x0)
print("hi")
for i in prange(settings.w * settings.h):
    print(i)
    x, y = i//settings.h, i%settings.h
    ray_dir = settings.ray_dir_px(x, y)
    V0 = settings.grav_field.null_cond(ray_dir, x0)
    y0 = np.concatenate((X0, V0, np.array([0], dtype=np.float64)))
    geodesics.append(trace_geodesic(integrator, y0, settings.bg_rad))
print("finished")
"""
