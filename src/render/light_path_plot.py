import numpy as np
import matplotlib.pyplot as plt

from render.Classes.base import *
from render.Classes.classes import GravField, RenderSettings

# Copied functions from rendering.py but without @njit decorator

def rk4_step(t, y, h, settings:RenderSettings):
    k1 = settings.geodesic_eq(t, y)
    k2 = settings.geodesic_eq(t+h/2, y+h*k1/2)
    k3 = settings.geodesic_eq(t+h/2, y+h*k2/2)
    k4 = settings.geodesic_eq(t+h, y+h*k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def integrator(settings:RenderSettings, y0:float, max_t:float=1000,
               tol:float=1e-8, h_init:float=0.1, safety:float=0.9):
    """Returns the point and the object the light ray hits."""
    t, y, h = 0., y0, h_init
    path = [y0]

    while t < max_t:
        
        # Check for any hits
        dists = np.empty(len(settings.scene), dtype=np.float64)
        for i in range(len(settings.scene)): # If the distance of closest approach is < 1e-4 or already inside
            obj = settings.scene[i]
            d = obj.shape.in_shape_int(settings.grav_field.mink_pos(y[:4]))
            if np.abs(d) < 1e-4: return path
            dists[i] = d
        if settings.escape(y) > 0: return path
        if h < 1e-4: return path

        # Adapt step size to how close the point is to any object
        speed = np.linalg.norm(settings.geodesic_eq(t, y))
        min_dist = min(dists)
        h = min(h, 1.25*min_dist/speed)
        if t + h > max_t: h = max_t - t # Prevents overstepping

        # Perform integration
        y_coarse = rk4_step(t, y, h, settings) # Coarse step
        y_mid = rk4_step(t, y, h/2, settings)
        y_fine = rk4_step(t+h/2, y_mid, h/2, settings) # Fine step made of 2 substeps
        # Error estimate
        error = np.max(np.abs(y_coarse - y_fine)) / 15
        if error <= tol:
            t += h
            y = y_fine
            path.append(y)
            if error < 1e-15 : h *= safety * 5.
            else:
                factor = (tol / error) ** (1/5)
                if factor < 0.1: h *= safety * 0.1
                elif 0.1 < factor < 5.0: h *= safety * factor
                else: h *= safety * 5.
        else: # Reject solution and reduce step size
            factor = (tol / error) ** (1/5)
            if factor > 0.1: h *= safety * factor
            else: h *= safety * 0.1

    return path

def plot(path:list, grav_field:GravField):
    x_plot, y_plot, z_plot = [], [], []
    for i in range(len(path)):
        mink_x = grav_field.mink_pos(path[i][:4])
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

def look_ray(ray_pos:Vec, ray_dir:Vec, settings:RenderSettings):
    ray_dir.is_normal()
    X0 = settings.grav_field.coord_pos(np.array([0, ray_pos.x, ray_pos.y, ray_pos.z]))
    V0 = settings.grav_field.null_cond(ray_dir, np.array([0, ray_pos.x, ray_pos.y, ray_pos.z]))
    y0 = np.array([X0[0], X0[1], X0[2], X0[3], V0[0], V0[1], V0[2], V0[3]])
    path = integrator(settings, y0)
    plot(path, settings.grav_field)
