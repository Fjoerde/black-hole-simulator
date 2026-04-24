import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from PIL import Image

from render.Classes.base import *
from render.Classes.tags import *
from render.Classes.classes import def_cc, Integrator, GravField, RenderSettings

def look_ray(ray_pos:Vec, ray_dir:Vec, settings:RenderSettings):
    """For plotting out the path of a light ray originating from ray_pos in the direction of ray_dir."""

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

def check_Gamma(x:np.ndarray, grav_field:GravField, h:float=1e-12) -> np.ndarray:
    """For verifying that the typed-out Christoffel symbol function is correct. Not used for rendering.
        
    x: A spacetime point described in metric coordinates."""

    dg = np.zeros((4,4,4), dtype=np.float64) # dg[k,i,j] is derivative of g[i,j] wrt x[k]
    for k in range(4):
        xp = x.copy(); xp[k] += h
        xm = x.copy(); xm[k] -= h
        gp = grav_field.sample_g(xp); gm = grav_field.sample_g(xm)
        dg[k] = (gp - gm) / (2 * h)
    real_Gamma = np.zeros((4,4,4), dtype=np.float64)
    inv_g = np.linalg.inv(grav_field.sample_g(x))
    for c in range(4):
        for a in range(4):
            for b in range(4):
                s = 0.
                for d in range(4):
                    s += dg[a,b,d] + dg[b,a,d] - dg[d,a,b]
                real_Gamma[c,a,b] = inv_g[c,d]/2 * s    
    diff = real_Gamma - grav_field.sample_Gamma(x)
    return diff

def plot_func(func, min:np.ndarray, max:np.ndarray):
    """Plot a function."""

    if func.dim not in [1,2]: raise ValueError("Function must only depend on 1 or 2 variables.")
    if func.entries != 1: raise ValueError("Function must only have 1 entry.")
    if not (min.shape[0] == max.shape[0] == func.dim): raise ValueError("Invalid shape for min and max.")

    if func.dim == 1:
        xx = np.linspace(min[0], max[0], 1000)
        plt.plot(xx, func.interp(xx.reshape(xx.shape[0], 1)), label="Function")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend()
        plt.show()
    else:
        X = np.linspace(min[0], max[0], 250); Y = np.linspace(min[1], max[1], 250)
        xx, yy = np.meshgrid(X, Y); zz = func.interp(Patch([X,Y]).pts).reshape(250, 250)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(xx, yy, zz, cmap="YlOrRd", label="Function")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        plt.legend()
        plt.tight_layout()
        plt.show()

def test_func(func:Function, ref, min:np.ndarray, max:np.ndarray):
    """Test the accuracy of a function given by its approximation (func) with its actual expression (ref).
    For a 1D function, plot both the reference and the interpolated functions.
    For a 2D function, plot the absolute difference between both."""
    
    if func.dim not in [1,2]: raise ValueError("Function must only depend on 1 or 2 variables.")
    if func.entries != 1: raise ValueError("Function must only have 1 entry.")
    if not (min.shape[0] == max.shape[0] == func.dim): raise ValueError("Invalid shape for min and max.")
    if not callable(ref): raise ValueError("ref must be a callable.")

    if func.dim == 1:
        xx = np.linspace(min[0], max[0], 1000)
        plt.plot(xx, ref(xx), label="Reference")
        plt.plot(xx, func.interp(xx.reshape(xx.shape[0], 1)), label="Interpolated")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend()
        plt.show()
    else:
        X = np.linspace(min[0], max[0], 250); Y = np.linspace(min[1], max[1], 250)
        xx, yy = np.meshgrid(X, Y); zz_ref = ref(xx, yy)
        zz_interp = func.interp(Patch([X,Y]).pts).reshape(250, 250)
        diff = np.abs(zz_ref - zz_interp)

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(xx, yy, diff, cmap="YlOrRd", label="Difference")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        plt.legend()
        plt.tight_layout()
        plt.show()

def look_col(spec_int:Function, col_converter:ColConverter=def_cc):
    rgb = col_converter.get_rgb(spec_int)
    img = np.empty((100, 100, 3))
    for i,j in product(range(100), range(100)): img[i,j] = rgb * 255
    img = Image.fromarray(img.astype(np.uint8))
    img.show()
    return rgb

def look_specint(col:np.ndarray, col_converter:ColConverter=def_cc, lux:float=-1.):
    spec_int = col_converter.get_spec_int(col, lux)
    min, max = np.min(spec_int.grid.pts), np.max(spec_int.grid.pts)
    xx = np.linspace(min, max, 1000)
    plt.plot(xx, spec_int.interp(xx.reshape(xx.shape[0], 1)), label="Spectral Intensity")
    plt.xlabel(r"$\lambda~/~\mathrm{nm}$"); plt.ylabel(r"$I_{\lambda}~/~\mathrm{W~m^{-2}~nm^{-1}}$")
    plt.legend()
    plt.show()
    return spec_int

