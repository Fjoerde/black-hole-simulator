import numpy as np
import matplotlib.pyplot as plt

from Classes.math import *
from Classes.physics import *
from Classes.int_and_settings import *

def get_obj_path(grav_field:GravField, init_pos:Vec, init_vel:Vec, t:float=0, max_tau:float=400, show=False) -> Function:
    """Obtain and view the path of the camera in a gravitational field given an initial position and velocity."""

    x0 = init_pos.four_vec(t)
    X0 = grav_field.coord_pos(x0)
    V0 = grav_field.timelike_cond(init_vel, x0)
    y0 = np.concatenate((X0, V0))
    integrator = Integrator(tag=INTEGRATOR_OBJPATH, grav_field=grav_field)
    geodesic = integrator.solve(0, y0, 5, max_tau, 1e-4)

    gdsic_vals = np.empty((len(geodesic.grid.pts), 8), dtype=np.float64)
    for i in range(len(geodesic.grid.pts)):
        x0 = geodesic.vals[i,:4]
        gdsic_vals[i,:4] = integrator.grav_field.mink_pos(x0)
        gdsic_vals[i,4:8] = integrator.grav_field.mink_vel(geodesic.vals[i,4:], x0)
    x_plot = gdsic_vals[:,1]; y_plot = gdsic_vals[:,2]; z_plot = gdsic_vals[:,3]

    if show:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot3D(x_plot, y_plot, z_plot, color="tab:blue")
        ax.plot3D(x_plot[0], y_plot[0], z_plot[0], "x", color="tab:orange", label="Start")
        ax.plot3D(x_plot[-1], y_plot[-1], z_plot[-1], "x", color="tab:green", label="End")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.grid(False)
        ax.set_title("Motion Path")
        plt.legend()
        plt.tight_layout()
        plt.show()

    new_geodesic = Function(geodesic.grid, np.ascontiguousarray(gdsic_vals))
    return new_geodesic

def look_to_origin(cam_worldline:Function, tau_scale:float, fps:float, frame_num:int) -> np.ndarray:
    """Obtain the yaw-pitch-roll angles for each frame given the camera's worldline and other video parameters,
    assuming that the camera always points towards the origin, and does not have a roll."""

    delta_tau = (1 / tau_scale) / fps
    taus = np.ascontiguousarray(np.arange(0, delta_tau*frame_num, delta_tau, dtype=np.float64))
    cam_four_pos = cam_worldline.interp(taus.reshape(frame_num, 1))
    xs = cam_four_pos[:,1:4]; rots = np.zeros((frame_num, 3), dtype=np.float64)
    for i in range(frame_num):
        cam_dir = -Vec(xs[i,0], xs[i,1], xs[i,2]).normal()
        theta = np.arccos(cam_dir.z); phi = np.arctan2(cam_dir.y, cam_dir.x)
        rots[i,0] = phi; rots[i,1] = theta
    return np.ascontiguousarray(rots)

