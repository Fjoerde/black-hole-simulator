import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from PIL import Image

from Classes.math import *
from Classes.tags import *
from Classes.physics import GravField
from Classes.int_and_settings import Integrator, RenderSettings
from render.img_rendering import *


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
    for c,a,b in product(range(4), repeat=3):
        s = 0.
        for d in range(4): s += dg[a,b,d] + dg[b,a,d] - dg[d,a,b]
        real_Gamma[c,a,b] = inv_g[c,d]/2 * s    
    diff = real_Gamma - grav_field.sample_Gamma(x)
    return diff


def plot_func(func, min:np.ndarray, max:np.ndarray, label:str="Function"):
    """Plot a function."""

    if func.dim not in [1,2]: raise ValueError("Function must only depend on 1 or 2 variables.")
    if func.entries != 1: raise ValueError("Function must only have 1 entry.")
    if not (min.shape[0] == max.shape[0] == func.dim): raise ValueError("Invalid shape for min and max.")

    if func.dim == 1:
        xx = np.linspace(min[0], max[0], 1000)
        plt.plot(xx, func.interp(xx.reshape(xx.shape[0], 1)[:,0]), label=label)
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend()
        plt.show()
    else:
        X = np.linspace(min[0], max[0], 250); Y = np.linspace(min[1], max[1], 250)
        xx, yy = np.meshgrid(X, Y); zz = func.interp(Patch([X,Y]).pts).reshape(250, 250)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(xx, yy, zz, cmap="YlOrRd", label=label)
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


def display_col(col:np.ndarray):
    """Displays a given RGB color."""

    img = np.empty((100, 100, 3))
    for i,j in product(range(100), range(100)): img[i,j] = col * 255
    img = Image.fromarray(img.astype(np.uint8))
    img.show()


def display_specint(spec_int:Function):
    """Displays a given spectral intensity."""

    min, max = np.min(spec_int.grid.pts), np.max(spec_int.grid.pts)
    xx = np.linspace(min, max, 1000)
    plt.plot(xx, spec_int.interp(xx.reshape(xx.shape[0], 1))[:,0], label="Spectral Intensity")
    plt.xlabel(r"$\lambda~/~\mathrm{nm}$"); plt.ylabel(r"$I_{\lambda}~/~\mathrm{W~m^{-2}~nm^{-1}}$")
    plt.legend()
    plt.show()


def bb_specint(temp:float, grid=Grid(Patch([np.linspace(380, 780, 41)]))) -> Function:
    """Returns the spectral intensity of a blackbody at a given temperature."""

    wvls = grid.pts.reshape(len(grid.pts)) * 1e-9
    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23
    spec_vals = 2*h*c**2/wvls**5 * 1/(np.exp((h*c)/(wvls*k*temp)) - 1) * 1e-9
    spec_int = Function(grid, spec_vals.reshape(len(spec_vals), 1))
    return spec_int


def look_ray(ray_pos:Vec, ray_dir:Vec, settings:RenderSettings) -> Function:
    """For plotting out the path of a light ray originating from ray_pos in the direction of ray_dir."""

    ray_dir.is_normal()
    x0 = ray_pos.four_vec(0)
    X0 = settings.grav_field.coord_pos(x0)
    V0 = settings.grav_field.null_cond(ray_dir, x0)
    y0 = np.concatenate((X0, V0))
    integrator = Integrator(tag=INTEGRATOR_GEODESICEQ, grav_field=settings.grav_field, scene=settings.scene,
                            cam_pos=settings.cam_pos, bg_rad=settings.bg_rad)
    geodesic = trace_geodesic(integrator, y0, settings.bg_rad)

    # Plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x_plot = geodesic.vals[:,1]; y_plot = geodesic.vals[:,2]; z_plot = geodesic.vals[:,3]
    ax.plot3D(x_plot, y_plot, z_plot, color="tab:olive")
    ax.plot3D(x_plot[0], y_plot[0], z_plot[0], "x", color="tab:blue", label="Start")
    ax.plot3D(x_plot[-1], y_plot[-1], z_plot[-1], "x", color="tab:orange", label="End")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.grid(False)
    ax.set_title("Light Ray")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return geodesic


def display_gas_vals(geodesic:Function, gas:Function, n:int) -> Function:
    """Plot the gas values along the given geodesic."""

    max_t = np.max(geodesic.grid.pts)
    grid = Grid(Patch([np.linspace(0, max_t, n, dtype=np.float64)]))
    xs = geodesic.interp(grid.pts)[:,:4]; gas_vals = gas.interp(xs)
    temp = gas_vals[:,4].reshape(len(gas_vals), 1); temp_func = Function(grid, np.ascontiguousarray(temp))
    ext_coeff = gas_vals[:,5].reshape(len(gas_vals), 1); ext_coeff_func = Function(grid, np.ascontiguousarray(ext_coeff))
    plot_func(temp_func, np.array([0.]), np.array([max_t]), label="Temperature")
    plot_func(ext_coeff_func, np.array([0.]), np.array([max_t]), label="Extinction Coefficient")
    gas_val_func = Function(grid, gas_vals)
    return gas_val_func


def ray_col(geodesic:Function, gas_val:Function, vel:Vec, settings:RenderSettings) -> tuple[Function, np.ndarray]:
    """Returns the spectral intensity and corresponding RGB color as seen by an observer receiving a
    light ray following a geodesic.
    
    vel: The three-velocity of the observer in Cartesian coordinates."""

    if vel.length() >= 1: raise ValueError("vel must have a norm less than 1.")

    grid = settings.col_converter.grid
    cam_pos = geodesic.vals[-1,:4]
    integrator = Integrator(tag=INTEGRATOR_SPECINT, specint_grid=grid,
                            geodesic=geodesic, gas_val=gas_val, grav_field=settings.grav_field,
                            obs_vel=settings.grav_field.timelike_cond(vel, cam_pos))
    # Find initial spectral intensity
    spec_int0 = np.zeros(len(grid.pts), dtype=np.float64)
    x0 = geodesic.vals[0,:4]; pos = Vec(x0[1], x0[2], x0[3])
    k0 = geodesic.vals[0,4:]; k1 = geodesic.vals[-1,4:]
    if (pos - settings.cam_pos).length() >= settings.bg_rad: # If light ray hits the background
        theta = np.acos(pos.z / pos.length()); phi = np.atan2(pos.y, pos.x)
        spec_int0 = settings.sample_bg(theta, phi)
        spec_int0 = settings.doppler_spec(spec_int0, x0, k0, k1).vals
        spec_int0 = spec_int0.reshape(len(spec_int0))
    else: # Find which object the light ray hits
        for obj in settings.scene:
            if obj.shape.on_surface(pos):
                spec_int0 = obj.spec_int(pos)
                spec_int0 = settings.doppler_spec(spec_int0, x0, k0, k1).vals
                spec_int0 = spec_int0.reshape(len(spec_int0))
                break
    max_t = np.max(geodesic.grid.pts)
    spec_int = integrator.solve(0, spec_int0, max_t/50, max_t, 10).vals[-1]
    spec_int = settings.rel_aberr(Function(grid, spec_int.reshape(len(spec_int), 1)), k1)
    rgb = settings.col_converter.get_rgb(spec_int)
    return spec_int, rgb