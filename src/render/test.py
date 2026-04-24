# A place for testing out functions/methods that would not otherwise be directly called.
import numpy as np
from render.Classes.base import *
from render.Classes.classes import *
from render.Classes.tags import *
from render.diagnostics import *
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# look_specint(np.array([0.3, 0.5, 0.1]))
ray_dir = Vec(1,0,0); ray_pos = Vec(0,0,0)
print("hi")
settings = RenderSettings()
print("hi2")

print("hi")
ray_dir.is_normal()
X0 = settings.grav_field.coord_pos(np.array([0, ray_pos.x, ray_pos.y, ray_pos.z]))
V0 = settings.grav_field.null_cond(ray_dir, np.array([0, ray_pos.x, ray_pos.y, ray_pos.z]))
y0 = np.concatenate((X0, V0, [0]))
integrator = Integrator(tag=INTEGRATOR_GEODESICEQ, grav_field=settings.grav_field, scene=settings.scene, cam_pos=settings.cam_pos, gas=settings.gas)
print("hi2")
geodesic = integrator.solve(0, y0, h_init=settings.bg_rad/50, max_t=settings.bg_rad*5)
print("hi3")
# Plot
x_plot, y_plot, z_plot = [], [], []
for i in range(len(geodesic.grid.pts)):
    mink_x = settings.grav_field.mink_pos(geodesic.grid.vals[i][:4])
x_plot.append(mink_x[1]); y_plot.append(mink_x[2]); z_plot.append(mink_x[3])
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(x_plot, y_plot, z_plot, color="tab:olive")
ax.plot3D(x_plot[0], y_plot[0], z_plot[0], "x", color="tab:blue")
ax.plot3D(x_plot[-1], y_plot[-1], z_plot[-1], "x", color="tab:orange")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.grid(False)
ax.set_title("Light Ray")
plt.tight_layout()
plt.show()
