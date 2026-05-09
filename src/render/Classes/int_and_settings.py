import numpy as np
from numba import int64, float64
from numba.core import types
from numba.typed import List

from render.Classes.math import *
from render.Classes.physics import *
from render.Classes.tags import *

import render.Classes.Integrators.geodesiceq as GeodesicEq
import render.Classes.Integrators.specint as SpecInt

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
                   ("specint_grid", Grid.class_type.instance_type), ("geodesics", types.ListType(Function.class_type.instance_type)),
                   ("gas_params", types.ListType(Function.class_type.instance_type)), ("doppler_obs", float64[::1]), ("solve_idx", int64)]
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
                 cam_pos:Vec=Vec(0,0,0), bg_rad:float=20, gas:Function=def_gas, specint_grid:Grid=Grid(),
                 geodesics:list[Function]=[Function()], gas_params:list[Function]=[Function()],
                 obs_vel:np.ndarray=np.array([1,0,0,0], dtype=np.float64)):
    
        self.tag = tag
        if self.tag == INTEGRATOR_GEODESICEQ: GeodesicEq.init(self, grav_field, scene, cam_pos, bg_rad, gas)
        if self.tag == INTEGRATOR_SPECINT: SpecInt.init(self, specint_grid, geodesics, gas_params, grav_field, obs_vel)

    def derivative(self, t:float, y:np.ndarray) -> np.ndarray:
        """Returns the derivative of the state vector y."""

        if self.tag == INTEGRATOR_GEODESICEQ: return GeodesicEq.derivative(self, t, y)
        elif self.tag == INTEGRATOR_SPECINT: return SpecInt.derivative(self, t, y)
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
        else: return False
    
    def sample_func(self, t:float, y:np.ndarray) -> np.ndarray:
        """Sample a function along the solution to the differential equation."""
        
        if self.tag == INTEGRATOR_GEODESICEQ: return GeodesicEq.sample_func(self, t, y)
        elif self.tag == INTEGRATOR_SPECINT: return SpecInt.sample_func(self, t, y)
        else: return np.ascontiguousarray(np.zeros(1, dtype=np.float64))
    
    def max_step(self, t:float, y:np.ndarray, h:float) -> float:
        """Returns the maximum step the integrator is allowed to achieve. Threshold may depend on state vector, parameter value, and step size."""

        if self.tag == INTEGRATOR_GEODESICEQ: return GeodesicEq.max_step(self, t, y, h)
        elif self.tag == INTEGRATOR_SPECINT: return SpecInt.max_step(self, t, y, h)
        else: return np.inf

    def solve(self, t0:float, y0:np.ndarray,
              h_init:float=0.1, max_t:float=1e3, tol:float=1e-8,
              max_iter:int=100, safety:float=0.9) -> tuple[Function, Function]:
        """Solves the differential equation and returns the function representing it.
        Allows sampling of a function f(x,y) along the solution g(t) = f(x, y(x))."""

        t, y, h, iter = t0, y0, h_init, 0
        ts, ys = np.array([t0], dtype=np.float64), y0.reshape(1, y0.shape[0])
        gs = self.sample_func(t0, y0); gs = gs.reshape(1, gs.shape[0])
        while (t < max_t and iter < max_iter):

            # Set integration step
            if np.isnan(ys).any() or np.isnan(gs).any(): raise ValueError("Encountered NaN values.")
            if self.term_cond(t, y, h): break
            h = min(h, self.max_step(t, y, h))
            if t + h > max_t: h = max_t - t

            # Perform integration
            y_coarse = self.rk4_step(t, y, h) # Coarse step
            y_mid = self.rk4_step(t, y, h/2)
            y_fine = self.rk4_step(t+h/2, y_mid, h/2) # Fine step made of 2 substeps

            # Sample function
            g_mid = self.sample_func(t, y_mid)
            g0 = gs[-1]; g1 = self.sample_func(t, y_fine)
            dg1 = g1 - g0
            if len(ts) != 1: dg0 = (g1 - gs[-2]) / (h + ts[-1] - ts[-2]) * h
            else: dg0 = dg1.copy()
            a = 2*g0 - 2*g1 + dg0 + dg1
            b = -3*g0 + 3*g1 - 2*dg0 - dg1
            c = dg0; d = g0
            g_mid_est = a/8 + b/4 + c/2 + d # Hermite interpolation in the middle
            
            # Error estimate
            ode_error = np.max(np.abs(y_coarse - y_fine)) / 15
            g_error = np.max(np.abs(g_mid_est - g_mid))
            error = max(ode_error, g_error)
            if error <= tol: # Accept solution
                t += h; y = y_fine
                ts = np.append(ts, t)
                ys = np.vstack((ys, y.reshape(1, len(y))))
                gs = np.vstack((gs, g1.reshape(1, len(g1))))
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
            iter += 1

        sol = Function(Grid(Patch([ts])), ys)
        func = Function(Grid(Patch([ts])), gs)
        return sol, func


# Render Settings
spec_settings = [("w", int64), ("h", int64), ("aspect", float64),
                 ("cam_pos", Vec.class_type.instance_type), ("cam_dir", Vec.class_type.instance_type), ("cam_vel", Vec.class_type.instance_type),
                 ("cam_u", Vec.class_type.instance_type), ("cam_v", Vec.class_type.instance_type),
                 ("scene", types.ListType(Hittable.class_type.instance_type)),
                 ("background", float64[:,:,::1]), ("bg_rad", float64),
                 ("bg_specints", float64[:,::1]),
                 ("col_converter", ColConverter.class_type.instance_type), ("gas", Function.class_type.instance_type),
                 ("grav_field", GravField.class_type.instance_type)]
@jitclass(spec_settings)
class RenderSettings:
    """The settings of the rendered scene. Includes parameters like camera position and angle, scene, background image, etc.
    
    For a scene with no object, simply omit the scene argument, or pass a list of a Hittable with a null Shape.
    
    bg_rad: Radius of the background box.
    
    bg_cols, bg_specints: The unqiue colors and spectral intensities of the background image.
    Must be consistent with the color converter grid and the image. No check is done to ensure that colors are truly
    unique, no colors are missed, and that every color has occurred in the image."""

    def __init__(self, w:int=800, h:int=600, # Width, height of image
                 cam_pos:Vec=Vec(0,0,0), cam_dir:Vec=Vec(1,0,0), cam_vel:Vec=Vec(0,0,0), # Position, direction, and velocity of camera
                 scene:list[Hittable]=def_scene, bg_rad:float=20, # List of hittables in the scene
                 col_converter:ColConverter=def_cc, gas:Function=def_gas, grav_field:GravField=GravField(tag=GRAVFIELD_MINKOWSKI)):
       
        if cam_vel.length() >= 1: raise ValueError("Velocity of camera must be smaller than 1.")
        for obj in scene:
            if (obj.shape.pos - cam_pos).length() > bg_rad: raise ValueError("All objects must be within the radius of the background image.")

        self.w = w
        self.h = h
        self.cam_pos = cam_pos
        self.cam_dir = cam_dir
        self.cam_vel = cam_vel
        self.scene = List(scene)
        self.bg_rad = bg_rad
        self.col_converter = col_converter
        self.gas = gas
        self.grav_field = grav_field

        self.aspect = self.w / self.h
        self.cam_u = self.cam_dir.cross(Vec(0,0,1)).normal() # Need to deal with cam_dir being close to Vec(0,0,1)
        self.cam_v = self.cam_u.cross(self.cam_dir).normal()

    def ray_dir_px(self, x:int, y:int) -> Vec:
        """Returns the three-direction of the light ray corresponding to a pixel on the final rendered image."""

        u = (2*(x+0.5)/self.w - 1) * self.aspect # Viewport coordinates, varies from -aspect to +aspect
        v = 1 - 2*(y+0.5)/self.h # varies from -1 to +1
        ray_dir_p = (self.cam_dir + u*self.cam_u + v*self.cam_v).normal() # p stands for primed (in the moving camera's frame)
        beta = self.cam_vel.length(); gamma = 1 / np.sqrt(1-beta**2)
        if np.abs(beta) < 1e-4: return ray_dir_p

        n_vel = self.cam_vel.normal()
        Y = (ray_dir_p - ray_dir_p.dot(n_vel)*n_vel).normal()
        theta_p = np.arccos(ray_dir_p.dot(n_vel))
        theta = np.atan2(np.sin(theta_p), gamma*(np.cos(theta_p)-beta)) # Relativistic aberration
        ray_dir = np.cos(theta)*n_vel + np.sin(theta)*Y
        return ray_dir.normal()
    
    """
    def sample_bg(self, theta:float, phi:float) -> Function:
        u, v = (phi+np.pi)/(2*np.pi), theta/np.pi
        h, w, _ = self.background.shape
        x, y = int(u*(w-1)), int(v*(h-1)) # Pixel coordinate on the image
        spec_int_val = self.bg_specints[y,x]
        spec_int = Function(self.col_converter.grid, spec_int_val.reshape(len(spec_int_val), 1))
        return spec_int
    """