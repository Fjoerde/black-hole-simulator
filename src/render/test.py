# A place for testing out functions/methods that would not otherwise be directly called.
from PIL import Image
from Classes.base import *
from Classes.tags import *
from Classes.classes import Shape, Hittable, ColField, RenderSettings
from rendering import *
from numba import prange
import Classes.GravField.methods as GravMethod

scene = [Hittable(tag=HITTABLE_LIGHT,
         shape=Shape(pos=Vec(0,0,0), rot=(np.pi/2,-np.pi/20,0), tag=SHAPE_ANNULUS, r_in=2.5, r_out=5, height=0.1),
         col_field=ColField(tag=COLFIELD_CONSTCOL, col=Vec(1,1,1)))] # Accretion disc
bg = np.array(Image.open("Images/background2.jpg")) / 255.0
settings = RenderSettings(w=800, h=600, cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), scene=scene, background=bg, bg_rad=20, threshold=1e-6) # If no object, simply omit scene argument

col = trace(settings.cam_pos, settings.ray_dir_px(0,0), settings)
print(col.np_array())


"""
for i in prange(120000, 240000):
    col = trace(settings.cam_pos, settings.ray_dir_px(i//settings.h,i%settings.h), settings)
    print(i, col.np_array())
"""
