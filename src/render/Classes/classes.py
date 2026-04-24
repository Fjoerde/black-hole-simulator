import numpy as np
from numba import float64, int64
from numba.experimental import jitclass
from numba.core import types
from numba.typed import List

from render.Classes.base import *
from render.Classes.tags import *

import render.Classes.Shapes.null as ShapeNull
import render.Classes.Shapes.sphere as Sphere
import render.Classes.Shapes.cylinder as Cylinder
import render.Classes.Shapes.annulus as Annulus

import render.Classes.Hittables.null as HittableNull
import render.Classes.Hittables.blackbody as Blackbody
import render.Classes.Hittables.checkerboard as Checkerboard

import render.Classes.GravField.minkowski as Minkowski
import render.Classes.GravField.schwarzschild as Schwarzschild
import render.Classes.GravField.kerr as Kerr
import render.Classes.GravField.kerr_newman as KerrNewman

import render.Classes.Integrators.geodesiceq as GeodesicEq
import render.Classes.Integrators.specint as SpecInt

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

spec_shape = [# For all Shapes
              ("pos", Vec.class_type.instance_type), ("rot", types.Tuple((float64, float64, float64))),
              ("X", Vec.class_type.instance_type), ("Y", Vec.class_type.instance_type), ("Z", Vec.class_type.instance_type),
              ("tag", int64),
              
              # For Sphere and Cylinder
              ("radius", float64),

              # For Cylinder and Annulus
              ("height", float64),

              # For Annulus
              ("r_in", float64), ("r_out", float64)]
@jitclass(spec_shape)
class Shape:
    def __init__(self, pos:Vec, rot:tuple[float,float,float], tag:int, # Rotation is represented by the three Euler angles
                 radius:float=0, height:float=0, r_in:float=0, r_out:float=0): 
        self.pos = pos
        self.rot = rot
        self.tag = tag
        rot_matrix = np.array([[np.cos(rot[0])*np.cos(rot[2])-np.sin(rot[0])*np.cos(rot[1])*np.sin(rot[2]), np.sin(rot[0])*np.cos(rot[2])+np.cos(rot[0])*np.cos(rot[1])*np.sin(rot[2]), np.sin(rot[1])*np.sin(rot[2])],
                               [-np.cos(rot[0])*np.sin(rot[2])-np.sin(rot[0])*np.cos(rot[1])*np.cos(rot[2]), -np.sin(rot[0])*np.sin(rot[2])+np.cos(rot[0])*np.cos(rot[1])*np.cos(rot[2]), np.sin(rot[1])*np.cos(rot[2])],
                               [np.sin(rot[0])*np.sin(rot[1]), -np.cos(rot[0])*np.sin(rot[1]), np.cos(rot[1])]])
        self.X = rot_matrix[0,0]*Vec(1,0,0) + rot_matrix[0,1]*Vec(0,1,0) + rot_matrix[0,2]*Vec(0,0,1)
        self.Y = rot_matrix[1,0]*Vec(1,0,0) + rot_matrix[1,1]*Vec(0,1,0) + rot_matrix[1,2]*Vec(0,0,1)
        self.Z = rot_matrix[2,0]*Vec(1,0,0) + rot_matrix[2,1]*Vec(0,1,0) + rot_matrix[2,2]*Vec(0,0,1) # Local coordinate axes

        if self.tag == SHAPE_NULL: ShapeNull.init(self)
        if self.tag == SHAPE_SPHERE: Sphere.init(self, radius)
        if self.tag == SHAPE_CYLINDER: Cylinder.init(self, radius, height)
        if self.tag == SHAPE_ANNULUS: Annulus.init(self, r_in, r_out, height)

    def closest_pt(self, pt:Vec) -> Vec:
        """Returns the closest point on the surface from pt."""

        if self.tag == SHAPE_NULL: return ShapeNull.closest_pt(self, pt)
        if self.tag == SHAPE_SPHERE: return Sphere.closest_pt(self, pt)
        if self.tag == SHAPE_CYLINDER: return Cylinder.closest_pt(self, pt)
        if self.tag == SHAPE_ANNULUS: return Annulus.closest_pt(self, pt)

    def on_surface(self, pt:Vec) -> bool:
        """Returns if a given point is on the surface of the shape."""

        return (pt - self.closest_pt(pt)).length() < 1e-4
    
    def in_shape(self, pt:Vec) -> bool:
        """Returns if a given point is within the shape."""

        if self.tag == SHAPE_NULL: return ShapeNull.in_shape(self, pt)
        if self.tag == SHAPE_SPHERE: return Sphere.in_shape(self, pt)
        if self.tag == SHAPE_CYLINDER: return Cylinder.in_shape(self, pt)
        if self.tag == SHAPE_ANNULUS: return Annulus.in_shape(self, pt)

    def in_shape_int(self, x:Vec) -> float:
        """Returns the distance to the closest point on the shape. Negative if the point is in the shape.

           Used as an event function for numerical integration."""

        pt = Vec(x[1], x[2], x[3])
        close_pt = self.closest_pt(pt)
        dist = (pt - close_pt).length()
        return -dist if self.in_shape(pt) else dist

    def projection(self, pt:Vec) -> tuple[float, float]:
        """Projects the surface of the shape onto a plane.

           Takes in a point on the surface of the shape and outputs its corresponding coordinate in the domain [0, 1]x[0, 1]."""
        
        if not self.on_surface(pt): raise ValueError("Expected a point on the surface of the shape.")

        if self.tag == SHAPE_NULL: return ShapeNull.projection(self, pt)
        if self.tag == SHAPE_SPHERE: return Sphere.projection(self, pt)
        if self.tag == SHAPE_CYLINDER: return Cylinder.projection(self, pt)
        if self.tag == SHAPE_ANNULUS: return Annulus.projection(self, pt)


# Hittables
spec_hittable = [# For all Hittables
                 ("shape", Shape.class_type.instance_type), ("tag", int64),

                 # For Blackbody
                 ("temp", float64),
                 
                 # For Checkerboard
                 ("spec_int1", Function.class_type.instance_type), ("spec_int2", Function.class_type.instance_type), ("n", int64),
                 ("col_converter", ColConverter.class_type.instance_type)]
@jitclass(spec_hittable)
class Hittable:
    def __init__(self, shape:Shape, tag:int,
                 temp:float=0.,
                 col1:np.ndarray=np.zeros(3, dtype=np.float64), col2:np.ndarray=np.zeros(3, dtype=np.float64), n:int=10,
                 col_converter:ColConverter=def_cc):
        self.shape = shape
        self.tag = tag

        if self.tag == HITTABLE_NULL: HittableNull.init(self)
        if self.tag == HITTABLE_BLACKBODY: Blackbody.init(self, temp)
        if self.tag == HITTABLE_CHECKERBOARD: Checkerboard.init(self, col1, col2, n, col_converter)

    def spec_int(self, pt) -> Function:
        """Returns the spectral intensity emitted by the Hittable at a pt."""

        if not self.shape.on_surface(pt): raise ValueError("Expected a point on the surface of the shape.")

        if self.tag == HITTABLE_NULL: return HittableNull.spec_int(self, pt)
        if self.tag == HITTABLE_BLACKBODY: return Blackbody.spec_int(self, pt)
        if self.tag == HITTABLE_CHECKERBOARD: return Checkerboard.spec_int(self, pt)


# Gases
null_4d_grid = Grid((Patch([np.zeros(1, dtype=np.float64),
                            np.zeros(1, dtype=np.float64),
                            np.zeros(1, dtype=np.float64),
                            np.zeros(1, dtype=np.float64)])))
null_scal_func = Function(null_4d_grid)
null_vec_func = Function(null_4d_grid, np.zeros((1, 4), dtype=np.float64))
spec_gas = [("vel", Function.class_type.instance_type), ("temp", Function.class_type.instance_type),
            ("ext_coeff", Function.class_type.instance_type)]
@jitclass(spec_gas)
class Gas:
    """Wrapper class for a gas that permeated the background box of the scene.
    
    vel: The proper velocity of the gas in Minkowski coordinates in terms of Minkowski coordinates, with respect to an observer at infinity.
    
    temp, ext_coeff: The temperature and extinction coefficient of the gas given in Minkowski coordinates."""

    def __init__(self, vel:Function=null_vec_func, temp:Function=null_scal_func, ext_coeff:Function=null_scal_func):
        if vel.entries != 4: raise ValueError("vel must have 4 entries.")
        if temp.entries != 1: raise ValueError("temp must have 1 entry.")
        if ext_coeff.entries != 1: raise ValueError("ext_coeff must have 1 entry.")

        self.vel = vel
        self.temp = temp
        self.ext_coeff = ext_coeff


# Gravitational Fields
spec_gravfield = [# For all GravFields
                  ("tag", int64),

                  # For Schwarzschild, Kerr, and Kerr-Newman
                  ("pos", Vec.class_type.instance_type), ("M", float64),

                  # For Kerr and Kerr-Newman
                  ("rot", float64[:,:]), ("rot_inv", float64[:,:]), ("J", float64),
                  
                  # For Kerr-Newman
                  ("Q", float64)]
@jitclass(spec_gravfield)
class GravField:
    def __init__(self, tag:int,
                 pos:Vec=Vec(0,0,0), M:float=0.,
                 ax:Vec=Vec(0,0,1), J:float=0.,
                 Q:float=0.):
        self.tag = tag

        ax.is_normal()
        if M < 0 or J < 0: raise ValueError("M and J must be non-negative. If J is negative, flip the axis.")
        if J**2 > M**2 * (M**2 - Q**2): raise ValueError("Black hole exhibits a naked singularity.")

        if self.tag == GRAVFIELD_MINKOWSKI: Minkowski.init(self)
        if self.tag == GRAVFIELD_SCHWARZSCHILD: Schwarzschild.init(self, pos, M)
        if self.tag == GRAVFIELD_KERR: Kerr.init(self, pos, M, ax, J)
        if self.tag == GRAVFIELD_KERRNEWMAN: KerrNewman.init(self, pos, M, ax, J, Q)
    
    def sample_g(self, x:np.ndarray) -> np.ndarray:
        """Returns the metric tensor at x."""

        if self.tag == GRAVFIELD_MINKOWSKI: return Minkowski.sample_g(self, x)
        if self.tag == GRAVFIELD_SCHWARZSCHILD: return Schwarzschild.sample_g(self, x)
        if self.tag == GRAVFIELD_KERR: return Kerr.sample_g(self, x)
        if self.tag == GRAVFIELD_KERRNEWMAN: return KerrNewman.sample_g(self, x)

    def sample_Gamma(self, x:np.ndarray) -> np.ndarray:
        """Returns the Christoffel symbols at x."""

        if self.tag == GRAVFIELD_MINKOWSKI: return Minkowski.sample_Gamma(self, x)
        if self.tag == GRAVFIELD_SCHWARZSCHILD: return Schwarzschild.sample_Gamma(self, x)
        if self.tag == GRAVFIELD_KERR: return Kerr.sample_Gamma(self, x)
        if self.tag == GRAVFIELD_KERRNEWMAN: return KerrNewman.sample_Gamma(self, x)
    
    def coord_pos(self, x:np.ndarray) -> np.ndarray:
        """x: A spacetime event described in Minkowski coordinates (relative to a stationary observer at infinity).

        Performs a coordinate transformation from Minkowski coordinates to that of the gravitational field's metric."""

        if self.tag == GRAVFIELD_MINKOWSKI: return Minkowski.coord_pos(self, x)
        elif self.tag == GRAVFIELD_SCHWARZSCHILD: return Schwarzschild.coord_pos(self, x)
        elif self.tag == GRAVFIELD_KERR: return Kerr.coord_pos(self, x)
        elif self.tag == GRAVFIELD_KERRNEWMAN: return KerrNewman.coord_pos(self, x)
        else: return np.zeros(4, dtype=np.float64)
    
    def mink_pos(self, x:np.ndarray) -> np.ndarray:
        """x: A spacetime event described in the coordiantes of the metric (relative to a stationary observer at infinity).

        Performs a coordinate transformation from the gravitational field's metric to Minkowski coordinates."""

        if self.tag == GRAVFIELD_MINKOWSKI: return Minkowski.mink_pos(self, x)
        elif self.tag == GRAVFIELD_SCHWARZSCHILD: return Schwarzschild.mink_pos(self, x)
        elif self.tag == GRAVFIELD_KERR: return Kerr.mink_pos(self, x)
        elif self.tag == GRAVFIELD_KERRNEWMAN: return KerrNewman.mink_pos(self, x)
        else: return np.zeros(4, dtype=np.float64)
    
    def jacobian(self, x:np.ndarray) -> np.ndarray:
        """x: A spacetime event described in Minkowski coordinates (relative to a stationary observer at infinity).
        
        Returns the Jacobian matrix of the coordinate transformation at x."""

        if self.tag == GRAVFIELD_MINKOWSKI: return Minkowski.jacobian(self, x)
        if self.tag == GRAVFIELD_SCHWARZSCHILD: return Schwarzschild.jacobian(self, x)
        if self.tag == GRAVFIELD_KERR: return Kerr.jacobian(self, x)
        if self.tag == GRAVFIELD_KERRNEWMAN: return KerrNewman.jacobian(self, x)

    def jacobian_inv(self, x:np.ndarray) -> np.ndarray:
        """x: A spacetime event described in the coordinates of the metric (relative to a stationary observer at infinity).
        
        Returns the inverse Jacobian matrix of the coordinate transformation at x."""

        if self.tag == GRAVFIELD_MINKOWSKI: return Minkowski.jacobian_inv(self, x)
        if self.tag == GRAVFIELD_SCHWARZSCHILD: return Schwarzschild.jacobian_inv(self, x)
        if self.tag == GRAVFIELD_KERR: return Kerr.jacobian_inv(self, x)
        if self.tag == GRAVFIELD_KERRNEWMAN: return KerrNewman.jacobian_inv(self, x)

    def coord_vel(self, v:np.ndarray, x:np.ndarray) -> np.ndarray:
        """v: A contravariant four-vector described in Minkowski coordinates (relative to a stationary observer at infinity).

        x: A spacetime point described in Minkowski coordinates (relative to a stationary observer at infinity).

        Performs a contravariant transformation of v at x from Minkowski coordinates to that of the gravitational field's metric."""

        """
        J = self.jacobian(x); V = np.zeros(4, dtype=np.float64)
        for i in range(4):
            for j in range(4): V[i] += J[i,j] * v[j]
        """
        return self.jacobian(x) @ v
    
    def mink_vel(self, v:np.ndarray, x:np.ndarray) -> np.ndarray:
        """v: A contravariant four-vector described in metric coordinates (relative to a stationary observer at infinity).

        x: A spacetime point described in metric coordinates (relative to a stationary observer at infinity).

        Performs a contravariant transformation of v at x from metric coordinates to Minkowski coordinates."""

        """
        J = self.jacobian_inv(x); V = np.zeros(4, dtype=np.float64)
        for i in range(4):
            for j in range(4): V[i] += J[i,j] * v[j]
        """
        return self.jacobian_inv(x) @ v
    
    def null_cond(self, V:Vec, x:np.ndarray) -> np.ndarray:
        """v: A three-vector described in Cartesian coordinates (relative to a stationary observer at infinity).

        x: A spacetime point described in Minkowski coordinates (relative to a stationary observer at infinity).
    
        Returns the normed vector (norm=0) at x in metric coordinates whose spatial components are the transformed components of vec."""

        J = self.jacobian(x); g = self.sample_g(self.coord_pos(x))
        k = (g @ J) @ J
        v = np.array([0, V.x, V.y, V.z])
        a = k[0,0]; b = 2 * (k[0,1:] @ v[1:]); c = (k[1:,1:] @ v[1:]) @ v[1:]
        """
        k = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                for mu in range(4):
                    for nu in range(4): k[i,j] += g[mu,nu] * J[mu,i] * J[nu,j]
    
        v = np.array([0, V.x, V.y, V.z])
        a = k[0,0]; b = c = 0
        for i in range(1,4): b += 2 * k[0,i] * v[i]
        for i in range(1,4):
            for j in range(4): c += k[i,j] * v[i] * v[j]
        """
        disc = b**2 - 4*a*c
        v[0] = min((-b-np.sqrt(disc))/(2*a), (-b+np.sqrt(disc))/(2*a))

        return self.coord_vel(v, x)    

    def timelike_cond(self, V:Vec, x:np.ndarray) -> np.ndarray:
        J = self.jacobian(x); g = self.sample_g(self.coord_pos(x))
        k = (g @ J) @ J
        v = np.array([0, V.x, V.y, V.z])
        a = k[0,0]; b = 2 * (k[0,1:] @ v[1:]); c = (k[1:,1:] @ v[1:]) @ v[1:] + 1
        disc = b**2 - 4*a*c
        v[0] = max((-b-np.sqrt(disc))/(2*a), (-b+np.sqrt(disc))/(2*a))
        return v      


# Integrator
default_scene = [Hittable(tag=HITTABLE_NULL,
                 shape=Shape(pos=Vec(0,0,0), rot=(0,0,0), tag=SHAPE_NULL))]
spec_integrator = [("tag", int64),

                   # For GeodesicEqs
                   ("scene", types.ListType(Hittable.class_type.instance_type)), ("cam_pos", Vec.class_type.instance_type),
                   
                   # For GeodesicEqs and SpecInts
                   ("grav_field", GravField.class_type.instance_type), ("gas", Gas.class_type.instance_type),

                   # For SpecInts
                   ("geodesic", Function.class_type.instance_type), ("specint_grid", Grid.class_type.instance_type),
                   ("doppler_obs", float64)]
@jitclass(spec_integrator)
class Integrator:
    """Initiate an integrator that solves differential equations.
    
    INTEGRATOR_GEODESICEQ solves for a geodesic given an initial state vector (x,v) and stops when it hits an object in the scene
    or escapes the background box.
    
    INTEGRATOR_SPECINT solves for a spectral intensity received by an observer of a light ray following a geodesic. The parameters of the gas
    are given over the geodesic which is parametrized by the unit interval. The background gravitational field and the observer's four-velocity
    need to be specified to account for the Doppler effect and gravitational redshift.
    
    specint_grid: A grid over which the spectral intensity values are tracked and solved over.
    
    geodesic: A function of 8 entries on the unit interval that parametrizes the geodesic. The first 4 entries are the four-positions, and the last
    4 are the scaled four-velocities.
    
    obs_vel: The four-velocity of the observer. No check is done to ensure that it is time-like."""

    def __init__(self, tag:int,
                 grav_field:GravField=GravField(tag=GRAVFIELD_MINKOWSKI), scene:list[Hittable]=default_scene, cam_pos:Vec=Vec(0,0,0), gas:Gas=Gas(),
                 specint_grid:Grid=Grid(), geodesic:Function=Function(), obs_vel:np.ndarray=np.array([-1,0,0,0], dtype=np.float64)):
    
        self.tag = tag
    
        if self.tag == INTEGRATOR_GEODESICEQ: GeodesicEq.init(self, grav_field, scene, cam_pos, gas)
        if self.tag == INTEGRATOR_SPECINT: SpecInt.init(self, specint_grid, geodesic, gas, grav_field, obs_vel)

    def derivative(self, t:float, y:np.ndarray) -> np.ndarray:
        """Returns the derivative of the state vector y."""

        if self.tag == INTEGRATOR_GEODESICEQ: return GeodesicEq.derivative(self, t, y)
        if self.tag == INTEGRATOR_SPECINT: return SpecInt.derivative(self, t, y)

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
        if self.tag == INTEGRATOR_SPECINT: return SpecInt.term_cond(self, t, y, h)
    
    def max_step(self, t:float, y:float, h:float) -> float:
        """Returns the maximum step the integrator is allowed to achieve. Threshold may depend on state vector, parameter value, and step size."""

        if self.tag == INTEGRATOR_GEODESICEQ: return GeodesicEq.max_step(self, t, y, h)
        if self.tag == INTEGRATOR_SPECINT: return SpecInt.max_step(self, t, y, h)

    def solve(self, t0:float, y0:np.ndarray,
              h_init:float=0.1, max_t:float=1e3, tol:float=1e-8, safety:float=0.9) -> Function:
        """Solves the differential equation and returns the function representing it."""

        t, y, h = t0, y0, h_init
        ts, ys = np.array([t0], dtype=np.float64), y0.reshape(1, y0.shape[0])
        while t < max_t:
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
                ts = np.concatenate((ts, t)); ys = np.vstack((ys, y))
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
        grid = Grid((Patch([ts])))
        return Function(grid, ys)


# Render Settings
spec_settings = [("w", int64), ("h", int64), ("aspect", float64),
                 ("cam_pos", Vec.class_type.instance_type), ("cam_dir", Vec.class_type.instance_type), ("cam_vel", Vec.class_type.instance_type),
                 ("cam_u", Vec.class_type.instance_type), ("cam_v", Vec.class_type.instance_type),
                 ("scene", types.ListType(Hittable.class_type.instance_type)),
                 ("background", float64[:,:,::1]), ("bg_rad", float64),
                 ("col_converter", ColConverter.class_type.instance_type), ("gas", Gas.class_type.instance_type),
                 ("grav_field", GravField.class_type.instance_type)]
@jitclass(spec_settings)
class RenderSettings:
    """The settings of the rendered scene. Includes parameters like camera position and angle, scene, background image, etc.
    
    For a scene with no object, simply omit the scene argument, or pass a list of a Hittable with a null Shape.
    
    bg_rad: Radius of the background box."""

    def __init__(self, w:int=800, h:int=600, # Width, height of image
                 cam_pos:Vec=Vec(0,0,0), cam_dir:Vec=Vec(1,0,0), cam_vel:Vec=Vec(0,0,0), # Position, direction, and velocity of camera
                 scene:list[Hittable]=default_scene, # List of hittables in the scene
                 background:np.ndarray=np.zeros((1,1,3)), bg_rad:float=100, # Background image (default black)
                 col_converter:ColConverter=def_cc, gas:Gas=Gas(), grav_field:GravField=GravField(tag=GRAVFIELD_MINKOWSKI)):
       
        if self.cam_vel.length() >= 1: raise ValueError("Velocity of camera must be smaller than 1.")
        for obj in scene:
            if (obj.shape.pos - cam_pos).length() > bg_rad: raise ValueError("All objects must be within the radius of the background image.")
       
        self.w = w
        self.h = h
        self.cam_pos = cam_pos
        self.cam_dir = cam_dir
        self.cam_vel = cam_vel
        self.scene = List(scene)
        self.background = background
        self.bg_rad = bg_rad
        self.col_converter = col_converter
        self.gas = gas
        self.grav_field = grav_field

        self.aspect = self.w / self.h
        self.cam_u = self.cam_dir.cross(Vec(0,0,1)).normal() # Need to deal with cam_dir being close to Vec(0,0,1)
        self.cam_v = self.cam_u.cross(self.cam_dir).normal()

    def ray_dir_px(self, x:int, y:int) -> Vec:
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
    
    def aberration(self, ray_dir:Vec) -> float:
        theta = np.arccos(ray_dir.dot(self.cam_vel.normal()))
        beta = self.cam_vel.length(); gamma = 1 / np.sqrt(1-beta**2)
        aberr = (gamma * (1+beta*np.cos(theta))) / (np.sin(theta)**2 + gamma**2*(np.cos(theta)+beta)**2)**1.5
        return aberr
    
    def sample_bg(self, theta:float, phi:float) -> np.ndarray:
        u, v = (phi+np.pi)/(2*np.pi), theta/np.pi
        h, w, _ = self.background.shape
        x, y = int(u*(w-1)), int(v*(h-1)) # Pixel coordinate on the image
        return self.background[y,x]