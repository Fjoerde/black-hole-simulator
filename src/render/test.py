# A place for testing out functions/methods that would not otherwise be directly called.

from Classes.classes import *
from rendering import *

scene = [Hittable(tag=HITTABLE_LIGHT,
         shape=Shape(pos=Vec(0,0,0), rot=(np.pi/2,-np.pi/10,0), tag=SHAPE_ANNULUS, r_in=1.5, r_out=5., height=0.15),
         col_field=ColField(tag=COLFIELD_CONSTCOL, col=Vec(1,1,1)))] # Accretion disc
bg = np.array(Image.open("Images/background2.jpg")) / 255.0
settings = RenderSettings(w=800, h=600, cam_pos=Vec(-10,0,0), cam_dir=Vec(1,0,0), scene=scene, background=bg, bg_rad=20, threshold=1e-6) # If no object, simply omit scene argument

print("hi")

"""
for i in prange(220000, 240000):
    x, y = i//settings.h, i%settings.h
    ray_dir = settings.ray_dir_px(x, y)
    col = trace(settings.cam_pos, ray_dir, settings)
    print(i, col.np_array())
"""


col = trace(settings.cam_pos, settings.ray_dir_px(379, 385), settings)
print(col.np_array())

