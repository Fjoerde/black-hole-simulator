import numpy as np
from numba import int64, float64
from numba.core import types
from numba.typed import List

from Classes.math import *
from Classes.physics import *
from Classes.tags import *

import Classes.Integrators.geodesiceq as GeodesicEq
import Classes.Integrators.specint as SpecInt
import Classes.Integrators.obj_path as ObjPath

# Integrator
def_scene = [Hittable(tag=HITTABLE_NULL,
                 shape=Shape(pos=Vec(0,0,0), rot=(0,0,0), tag=SHAPE_NULL))]
spec_integrator = [("tag", int64),

                   # For GeodesicEqs
                   ("scene", types.ListType(Hittable.class_type.instance_type)), ("cam_pos", Vec.class_type.instance_type),
                   ("bg_rad", float64),
                   
                   # For GeodesicEqs and SpecInts
                   ("grav_field", GravField.class_type.instance_type), ("gas", Function.class_type.instance_type),

                   # For SpecInts
                   ("specint_grid", Grid.class_type.instance_type), ("geodesic", Function.class_type.instance_type),
                   ("gas_val", Function.class_type.instance_type), ("doppler_obs", float64)]
@jitclass(spec_integrator)
class Integrator:
    """Initiate an integrator that solves differential equations.
    
    INTEGRATOR_GEODESICEQ solves for a geodesic given an initial state vector (x,v) and stops when it hits an object in the scene
    or escapes the background box.
    
    INTEGRATOR_SPECINT solves for a spectral intensity received by an observer of a light ray following a geodesic. The background gravitational field and
    the observer's four-velocity need to be specified to account for the Doppler effect and gravitational redshift.
    
    specint_grid: A grid over which the spectral intensity values are tracked and solved over.
    
    geodesics: A list of function of 9 entries on the unit interval that parametrizes the geodesic. The first 4 entries are the four-positions, and the next
    4 are the scaled four-velocities, and the last entry is the extinction coefficient. To solve for the spectral intensity for a light ray following
    a specific geodesic in geodesics, set solve_idx to be the index of the desired geodesic.
    
    obs_vel: The four-velocity of the observer. No check is done to ensure that it is time-like."""

    def __init__(self, tag:int,
                 grav_field:GravField=GravField(tag=GRAVFIELD_MINKOWSKI), scene:list[Hittable]=def_scene,
                 cam_pos:Vec=Vec(0,0,0), bg_rad:float=20, specint_grid:Grid=Grid(),
                 geodesic:Function=Function(), gas_val:Function=Function(),
                 obs_vel:np.ndarray=np.array([1,0,0,0], dtype=np.float64)):
    
        self.tag = tag
        if self.tag == INTEGRATOR_GEODESICEQ: GeodesicEq.init(self, grav_field, scene, cam_pos, bg_rad)
        if self.tag == INTEGRATOR_SPECINT: SpecInt.init(self, specint_grid, geodesic, gas_val, grav_field, obs_vel)
        if self.tag == INTEGRATOR_OBJPATH: ObjPath.init(self, grav_field)

    def derivative(self, t:float, y:np.ndarray) -> np.ndarray:
        """Returns the derivative of the state vector y."""

        if self.tag == INTEGRATOR_GEODESICEQ: return GeodesicEq.derivative(self, t, y)
        elif self.tag == INTEGRATOR_SPECINT: return SpecInt.derivative(self, t, y)
        elif self.tag == INTEGRATOR_OBJPATH: return ObjPath.derivative(self, t, y)
        else: return np.zeros(1, dtype=np.float64)

    def rk4_step(self, t:float, y:np.ndarray, h:float) -> np.ndarray:
        """Returns the state vector at a step h away with the Runge-Kutta method."""

        k1 = self.derivative(t, y)
        k2 = self.derivative(t+h/2, y+h*k1/2)
        k3 = self.derivative(t+h/2, y+h*k2/2)
        k4 = self.derivative(t+h, y+h*k3)
        return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    def term_cond(self, t:float, y:np.ndarray, h:float) -> bool:
        """Returns if the integration is to be terminated early based on state vector, parameter value, and step size."""

        if self.tag == INTEGRATOR_GEODESICEQ: return GeodesicEq.term_cond(self, t, y, h)
        elif self.tag == INTEGRATOR_SPECINT: return SpecInt.term_cond(self, t, y, h)
        elif self.tag == INTEGRATOR_OBJPATH: return ObjPath.term_cond(self, t, y, h)
        else: return False
    
    def max_step(self, t:float, y:np.ndarray, h:float) -> float:
        """Returns the maximum step the integrator is allowed to achieve. Threshold may depend on state vector, parameter value, and step size."""

        if self.tag == INTEGRATOR_GEODESICEQ: return GeodesicEq.max_step(self, t, y, h)
        elif self.tag == INTEGRATOR_SPECINT: return SpecInt.max_step(self, t, y, h)
        elif self.tag == INTEGRATOR_OBJPATH: return ObjPath.max_step(self, t, y, h)
        else: return np.inf

    def adapt_stepsize(self, h:float, error:float, tol:float) -> float:
        """Returns the adapted step size based on error and inputted tolerance."""

        if error <= tol:
            if error > 0: factor = (tol / error) ** (1/5)
            else: factor = 5.
            return h * min(5, factor)
        else:
            if error > 0: factor = (tol / error) ** (1/5)
            else: factor = 0.1
            return h * max(0.1, factor)

    def solve(self, t0:float, y0:np.ndarray,
              h_init:float=0.1, max_t:float=1e3, tol:float=1e-6, safety:float=0.5) -> Function:
        """Solves the differential equation and returns the function representing it."""

        t, y, h = t0, y0, h_init
        ts, ys = np.array([t0], dtype=np.float64), y0.reshape(1, y0.shape[0])
        while t < max_t:

            # Set integration step
            if np.isnan(ys).any(): raise ValueError("Encountered NaN values.")
            if self.term_cond(t, y, h): break
            h = min(h, self.max_step(t, y, h))
            if t + h > max_t: h = max_t - t

            # Perform integration
            y_coarse = self.rk4_step(t, y, h) # Coarse step
            y_mid = self.rk4_step(t, y, h/2)
            y_fine = self.rk4_step(t+h/2, y_mid, h/2) # Fine step made of 2 substeps
            
            # Error estimate
            error = np.max(np.abs(y_coarse - y_fine)) / 15
            if error <= tol: # Accept solution
                t += h; y = y_fine
                ts = np.append(ts, t)
                ys = np.vstack((ys, y.reshape(1, len(y))))
            h = safety * self.adapt_stepsize(h, error, tol)

        sol = Function(Grid(Patch([ts])), ys)
        return sol


# Render Settings
spec_settings = [("w", int64), ("h", int64), ("f", float64), ("t", float64), ("aspect", float64),
                 ("cam_pos", Vec.class_type.instance_type), ("cam_dir", Vec.class_type.instance_type), ("cam_vel", Vec.class_type.instance_type),
                 ("cam_u", Vec.class_type.instance_type), ("cam_v", Vec.class_type.instance_type),
                 ("scene", types.ListType(Hittable.class_type.instance_type)),
                 ("background", float64[:,:,::1]), ("bg_rad", float64), ("col_converter", ColConverter.class_type.instance_type),
                 ("gas", Function.class_type.instance_type), ("grav_field", GravField.class_type.instance_type)]
@jitclass(spec_settings)
class RenderSettings:
    """Wrapper class for the settings of the rendered scene. Includes parameters like camera position, angle, and scene, etc.
    
    For a scene with no object, simply omit the scene argument, or pass a list of a Hittable with a null Shape.

    rot: A tuple of three floats, specifying yaw, pitch, and roll respectively.
    
    bg_rad: Radius of the background box."""

    def __init__(self, w:int=800, h:int=600, f:float=1, # Width, height of image, focal length of camera
                 cam_pos:Vec=Vec(0,0,0), cam_vel:Vec=Vec(0,0,0), t:float=0, # Position and velocity of camera, t-coordinate
                 rot:tuple[float,float,float]=(0,np.pi/2,0), # Yaw, pitch, and roll
                 scene:list[Hittable]=def_scene, background:np.ndarray=np.zeros((1,1,3), dtype=np.float64), bg_rad:float=20, 
                 col_converter:ColConverter=def_cc, gas:Function=def_gas, grav_field:GravField=GravField(tag=GRAVFIELD_MINKOWSKI)):
       
        if gas.entries != 6: raise ValueError("Invalid number of entries for the gas.")
        if (background > 1).any(): raise ValueError("Encountered invalid background colors.")
        if cam_vel.length() >= 1: raise ValueError("Velocity of camera must be smaller than 1.")
        for obj in scene:
            if (obj.shape.pos - cam_pos).length() > bg_rad: raise ValueError("All objects must be within the radius of the background image.")

        phi, theta, _ = rot
        self.w = w
        self.h = h
        self.f = f
        self.t = t
        self.cam_pos = cam_pos
        self.cam_dir = Vec(np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta))
        self.cam_vel = cam_vel
        self.scene = List(scene)
        self.background = background
        self.bg_rad = bg_rad
        self.col_converter = col_converter
        self.gas = gas
        self.grav_field = grav_field

        self.aspect = self.w / self.h
        self.cam_u, self.cam_v = self.viewport_basis(rot)

    def viewport_basis(self, rot) -> tuple[Vec, Vec]:
        phi, _, xi = rot
        cam_u = Vec(np.sin(phi), -np.cos(phi), 0)
        X = cam_u; Y = self.cam_dir.cross(X)
        cam_u = np.cos(xi) * X + np.sin(xi) * Y; cam_v = -Y
        return cam_u, cam_v

    def ray_dir_px(self, x:int, y:int) -> Vec:
        """Returns the three-direction of the light ray corresponding to a pixel on the final rendered image."""

        u = (2*(x+0.5)/self.w - 1) * self.aspect # Viewport coordinates, varies from -aspect to +aspect
        v = 1 - 2*(y+0.5)/self.h # varies from -1 to +1
        ray_dir_p = (self.f*self.cam_dir + u*self.cam_u + v*self.cam_v).normal() # p stands for primed (in the moving camera's frame)
        beta = self.cam_vel.length(); gamma = 1 / np.sqrt(1-beta**2)
        if np.abs(beta) < 1e-4: return ray_dir_p

        n_vel = self.cam_vel.normal()
        Y = (ray_dir_p - ray_dir_p.dot(n_vel)*n_vel).normal()
        theta_p = np.arccos(ray_dir_p.dot(n_vel))
        theta = np.atan2(np.sin(theta_p), gamma*(np.cos(theta_p)-beta)) # Relativistic aberration
        ray_dir = np.cos(theta)*n_vel + np.sin(theta)*Y
        return ray_dir.normal()

    def sample_bg(self, theta:float, phi:float) -> Function:
        """Returns the spectral radiance corresponding to the color of the background at a specific angular coordinate."""

        u, v = (phi+np.pi)/(2*np.pi), theta/np.pi
        h, w, _ = self.background.shape
        x, y = int(u*(w-1)), int(v*(h-1)) # Pixel coordinate on the image
        spec_int = self.col_converter.get_spec_int(self.background[y,x])
        return spec_int
    
    def doppler_spec(self, spec_int:Function, x_src:np.ndarray, k_src:np.ndarray, k_obs:np.ndarray) -> Function:
        """Returns the Doppler-shifted spectrum emitted by a stationary object received by an observer.
        
        spec_int: The emitted spectral intensity.
        
        x_src: The four-position of the source in Minkowski coordinates.
        
        k_src, k_obs: The four-velocities of the geodesic at the source and observer."""

        J = self.grav_field.jacobian(x_src); g = self.grav_field.sample_g(self.grav_field.coord_pos(x_src))
        g = J.T @ g @ J
        D_src = (g @ k_src) @ np.array([1,0,0,0], dtype=np.float64)
        D_obs = (g @ k_obs) @ self.grav_field.timelike_cond(self.cam_vel, self.cam_pos.four_vec(self.t))
        if max(np.abs(D_obs), np.abs(D_src)) < 1e-16: D = 1
        else: D = D_src / D_obs

        spectrum = self.col_converter.grid.pts
        spec_min = np.min(spectrum); spec_max = np.max(spectrum)
        shift_spec = np.where((spectrum / D >= spec_min) & (spectrum / D <= spec_max), spec_int.interp(spectrum / D), 0)
        shift_spec /= D**5
        new_specint = Function(self.col_converter.grid, np.ascontiguousarray(shift_spec))
        return new_specint
    
    def rel_aberr(self, spec_int:Function, k_obs:np.ndarray):
        """Returns the spectral radiance received by an observer due to Doppler beaming along a geodesic.
        
        spec_int: The spectral radiance of the light ray at the end of the geodesic.
        
        k_obs: The four-velocity of the geodesic at the position of the observer."""

        if self.cam_vel.length() > 1e-16:
            ray_dir = -Vec(k_obs[1], k_obs[2], k_obs[3]).normal()
            theta = np.arccos(ray_dir.dot(self.cam_vel.normal()))
            beta = self.cam_vel.length(); gamma = 1 / np.sqrt(1-beta**2)
            aberr = (gamma * (1+beta*np.cos(theta))) / (np.sin(theta)**2 + gamma**2*(np.cos(theta)+beta)**2)**1.5
            spec_vals = spec_int.vals / aberr
            new_specint = Function(spec_int.grid, np.ascontiguousarray(spec_vals))
            return new_specint
        else: return spec_int


# Video Settings
def_worldline = Function(Grid(Patch([np.array([0], dtype=np.float64)])), np.array([[0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.float64))
def_rots = np.zeros((100, 3), dtype=np.float64); def_rots[:,1] = np.pi / 2
spec_vidsettings = [("w", int64), ("h", int64), ("f", float64), ("aspect", float64), ("tau_scale", float64),
                    ("cam_worldline", Function.class_type.instance_type), ("rots", float64[:,::1]),
                    ("fps", float64), ("frame_num", int64), ("output_path", types.unicode_type),
                    ("t", float64[::1]), ("cam_pos", types.ListType(Vec.class_type.instance_type)), ("cam_vel", types.ListType(Vec.class_type.instance_type)),
                    ("cam_dir", types.ListType(Vec.class_type.instance_type)), ("cam_u", types.ListType(Vec.class_type.instance_type)),
                    ("cam_v", types.ListType(Vec.class_type.instance_type)), ("scene", types.ListType(Hittable.class_type.instance_type)),
                    ("background", float64[:,:,::1]), ("bg_rad", float64), ("col_converter", ColConverter.class_type.instance_type),
                    ("gas", Function.class_type.instance_type), ("grav_field", GravField.class_type.instance_type)]
@jitclass(spec_vidsettings)
class VidSettings:
    def __init__(self, w:int=800, h:int=600, f:float=1, tau_scale:float=1, # 1 unit of proper time corresponds to how many seconds
                 cam_worldline:Function=def_worldline, rots:np.ndarray=def_rots, # Camera worldline parametrized by proper time
                 fps:float=30, frame_num:int=100, output_path:str="Images/video.mp4",
                 scene:list[Hittable]=def_scene, background:np.ndarray=np.zeros((1,1,3), dtype=np.float64),
                 bg_rad:float=20, col_converter:ColConverter=def_cc, gas:Function=def_gas, grav_field:GravField=GravField(tag=GRAVFIELD_MINKOWSKI)):
        
        if len(rots) != frame_num: raise ValueError("rots must have the same number of entries as frame numbers.")
        if rots.shape[1] != 3: raise ValueError("Invalid shape for rots.")
        if tau_scale <= 0: raise ValueError("tau_scale must be a positive number.")
        if fps <= 0: raise ValueError("fps must be a positive number.")
        if gas.entries != 6: raise ValueError("Invalid number of entries for the gas.")
        if (background > 1).any(): raise ValueError("Encountered invalid background colors.")
        if cam_worldline.vals.shape[1] != 8: raise ValueError("Invalid number of entries for the camera's worldline.")

        self.w = w
        self.h = h
        self.f = f
        self.tau_scale = tau_scale
        self.cam_worldline = cam_worldline
        self.rots = np.ascontiguousarray(rots)
        self.fps = fps
        self.frame_num = frame_num
        self.output_path = output_path
        self.scene = List(scene)
        self.background = background
        self.bg_rad = bg_rad
        self.col_converter = col_converter
        self.gas = gas
        self.grav_field = grav_field
        
        self.aspect = self.w / self.h
        self.t, self.cam_pos, self.cam_vel = self.get_frame_param()
        self.cam_dir, self.cam_u, self.cam_v = self.get_viewport_bases()

    def get_frame_param(self) -> tuple[np.ndarray[float], list[Vec], list[Vec]]:
        """Calculate the t-coordinates, positions, and velocities of the camera at each rendered frame."""

        delta_tau = (1 / self.tau_scale) / self.fps
        taus = np.ascontiguousarray(np.arange(0, delta_tau*self.frame_num, delta_tau, dtype=np.float64))
        cam_four_pos = self.cam_worldline.interp(taus.reshape(self.frame_num, 1))
        ts = cam_four_pos[:,0]; pos = []; vel = []
        for i in range(self.frame_num):
            pos.append(Vec(cam_four_pos[i,1], cam_four_pos[i,2], cam_four_pos[i,3]))
            cam_vel = cam_four_pos[i,5:] / cam_four_pos[i,4] # Transform back to ordinary velocity
            vel.append(Vec(cam_vel[0], cam_vel[1], cam_vel[2]))
        return np.ascontiguousarray(ts), List(pos), List(vel)
    
    def get_viewport_bases(self) -> tuple[list[Vec], list[Vec], list[Vec]]:
        """Calculate the camera direction and viewport basis vectors for each frame."""

        cam_dir, cam_u, cam_v = [], [], []
        for i in range(self.frame_num):
            phi, theta, xi = self.rots[i]
            cam_dir.append(Vec(np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)))
            X = Vec(np.sin(phi), -np.cos(phi), 0)
            Y = cam_dir[-1].cross(X)
            cam_u.append(np.cos(xi) * X + np.sin(xi) * Y)
            cam_v.append(-Y)
        return List(cam_dir), List(cam_u), List(cam_v)
    
    def ray_dir_px(self, x:int, y:int, frame_num:int) -> Vec:
        """Returns the three-direction of the light ray corresponding to a pixel on the final rendered image."""

        cam_dir, cam_u, cam_v = self.cam_dir[frame_num], self.cam_u[frame_num], self.cam_v[frame_num]
        cam_vel = self.cam_vel[frame_num]
        print(cam_dir.np_array()); print(cam_vel.np_array())
        u = (2*(x+0.5)/self.w - 1) * self.aspect # Viewport coordinates, varies from -aspect to +aspect
        v = 1 - 2*(y+0.5)/self.h # varies from -1 to +1
        ray_dir_p = (self.f*cam_dir + u*cam_u + v*cam_v).normal() # p stands for primed (in the moving camera's frame)
        beta = cam_vel.length(); gamma = 1 / np.sqrt(1-beta**2)
        if np.abs(beta) < 1e-4: return ray_dir_p

        n_vel = cam_vel.normal()
        Y = (ray_dir_p - ray_dir_p.dot(n_vel)*n_vel).normal()
        theta_p = np.arccos(ray_dir_p.dot(n_vel))
        theta = np.atan2(np.sin(theta_p), gamma*(np.cos(theta_p)-beta)) # Relativistic aberration
        ray_dir = np.cos(theta)*n_vel + np.sin(theta)*Y
        return ray_dir.normal()

    def sample_bg(self, theta:float, phi:float) -> Function:
        """Returns the spectral radiance corresponding to the color of the background at a specific angular coordinate."""

        u, v = (phi+np.pi)/(2*np.pi), theta/np.pi
        h, w, _ = self.background.shape
        x, y = int(u*(w-1)), int(v*(h-1)) # Pixel coordinate on the image
        spec_int = self.col_converter.get_spec_int(self.background[y,x])
        return spec_int
    
    def doppler_spec(self, spec_int:Function, x_src:np.ndarray, k_src:np.ndarray, k_obs:np.ndarray, frame_num:int) -> Function:
        """Returns the Doppler-shifted spectrum emitted by a stationary object received by an observer.
        
        spec_int: The emitted spectral intensity.
        
        x_src: The four-position of the source in Minkowski coordinates.
        
        k_src, k_obs: The four-velocities of the geodesic at the source and observer."""

        J = self.grav_field.jacobian(x_src); g = self.grav_field.sample_g(self.grav_field.coord_pos(x_src))
        g = J.T @ g @ J
        D_src = (g @ k_src) @ np.array([1,0,0,0], dtype=np.float64)
        D_obs = (g @ k_obs) @ self.grav_field.timelike_cond(self.cam_vel[frame_num], self.cam_pos[frame_num].four_vec(self.t[frame_num]))
        if max(np.abs(D_obs), np.abs(D_src)) < 1e-16: D = 1
        else: D = D_src / D_obs

        spectrum = self.col_converter.grid.pts
        spec_min = np.min(spectrum); spec_max = np.max(spectrum)
        shift_spec = np.where((spectrum / D >= spec_min) & (spectrum / D <= spec_max), spec_int.interp(spectrum / D), 0)
        shift_spec /= D**5
        new_specint = Function(self.col_converter.grid, np.ascontiguousarray(shift_spec))
        return new_specint
    
    def rel_aberr(self, spec_int:Function, k_obs:np.ndarray, frame_num:int):
        """Returns the spectral radiance received by an observer due to Doppler beaming along a geodesic.
        
        spec_int: The spectral radiance of the light ray at the end of the geodesic.
        
        k_obs: The four-velocity of the geodesic at the position of the observer."""

        if self.cam_vel[frame_num].length() > 1e-16:
            ray_dir = -Vec(k_obs[1], k_obs[2], k_obs[3]).normal()
            theta = np.arccos(ray_dir.dot(self.cam_vel[frame_num].normal()))
            beta = self.cam_vel[frame_num].length(); gamma = 1 / np.sqrt(1-beta**2)
            aberr = (gamma * (1+beta*np.cos(theta))) / (np.sin(theta)**2 + gamma**2*(np.cos(theta)+beta)**2)**1.5
            spec_vals = spec_int.vals / aberr
            new_specint = Function(spec_int.grid, np.ascontiguousarray(spec_vals))
            return new_specint
        else: return spec_int