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

import render.Classes.ColFields.const_col as ConstCol
import render.Classes.ColFields.checkerboard as Checkerboard

import render.Classes.Hittables.null as HittableNull
import render.Classes.Hittables.light as Light

import render.Classes.GravField.minkowski as Minkowski
import render.Classes.GravField.schwarzschild as Schwarzschild
import render.Classes.GravField.kerr as Kerr
import render.Classes.GravField.kerr_newman as KerrNewman

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Class
spec_shape = [# For all Shapes
              ("pos", Vec.class_type.instance_type), ("rot", types.Tuple((float64, float64, float64))),
              ("X", Vec.class_type.instance_type), ("Y", Vec.class_type.instance_type), ("Z", Vec.class_type.instance_type),
              ("bb", BoundingBox.class_type.instance_type), ("tag", int64),
              
              # For Sphere and Cylinder
              ("radius", float64),

              # For Cylinder and Annulus
              ("height", float64), ("c_top", Vec.class_type.instance_type), ("c_base", Vec.class_type.instance_type),

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
    

# Color Fields
spec_colfield = [# For any ColField
                 ("tag", int64),

                 # For ConstCol
                 ("col", Vec.class_type.instance_type),
                 
                 # For Checkerboard
                 ("col1", Vec.class_type.instance_type), ("col2", Vec.class_type.instance_type)]
@jitclass(spec_colfield)
class ColField: # Defined on an abstract uv-plane [0,1]x[0,1] that maps onto the surface of the shape
    def __init__(self, tag=int,
                 col:Vec=Vec(0,0,0),
                 col1:Vec=Vec(0,0,0), col2:Vec=Vec(0,0,0)):
        self.tag = tag

        if self.tag == COLFIELD_CONSTCOL: ConstCol.init(self, col)
        if self.tag == COLFIELD_CHECKERBOARD: Checkerboard.init(self, col1, col2)

    def sample(self, u:float, v:float) -> Vec:
        """Returns the color at that specific (u,v) coordiante."""

        if self.tag == COLFIELD_CONSTCOL: return ConstCol.sample(self, u, v)
        if self.tag == COLFIELD_CHECKERBOARD: return Checkerboard.sample(self, u, v)
        

# Hittables
spec_hittable = [# For all Hittables
                 ("shape", Shape.class_type.instance_type), ("tag", int64),

                 # For Light
                 ("col_field", ColField.class_type.instance_type)]
@jitclass(spec_hittable)
class Hittable:
    def __init__(self, shape:Shape, tag:int,
                 col_field:ColField=ColField(tag=COLFIELD_CONSTCOL, col=Vec(0,0,0))):
        self.shape = shape
        self.tag = tag

        if self.tag == HITTABLE_NULL: HittableNull.init(self)
        if self.tag == HITTABLE_LIGHT: Light.init(self, col_field)

    def color(self, pt):
        """Returns the color of the Hittable."""

        if not self.shape.on_surface(pt): raise ValueError("Expected a point on the surface of the shape.")

        if self.tag == HITTABLE_NULL: return HittableNull.color(self, pt)
        if self.tag == HITTABLE_LIGHT: return Light.color(self, pt)


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
class GravField: # In natural units, G = c = 1, epsilon_0 = 1/(4*pi)
    def __init__(self, tag:int,
                 pos:Vec=Vec(0,0,0), M:float=1e-15,
                 ax:Vec=Vec(0,0,1), J:float=0.,
                 Q:float=0.):
        self.tag = tag

        ax.is_normal()
        if M < 0 or J < 0: raise ValueError("M and J must be non-negative. If J is negative, flip the axis.")
        if (J/M)**2 + Q**2 > M**2: raise ValueError("Black hole exhibits a naked singularity.")

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

        J = self.jacobian(x); V = np.zeros(4, dtype=np.float64)
        for i in range(4):
            for j in range(4): V[i] += J[i,j] * v[j]
        return V
    
    def mink_vel(self, v:np.ndarray, x:np.ndarray) -> np.ndarray:
        """v: A contravariant four-vector described in metric coordinates (relative to a stationary observer at infinity).

        x: A spacetime point described in metric coordinates (relative to a stationary observer at infinity).

        Performs a contravariant transformation of v at x from metric coordinates to Minkowski coordinates."""

        J = self.jacobian_inv(x); V = np.zeros(4, dtype=np.float64)
        for i in range(4):
            for j in range(4): V[i] += J[i,j] * v[j]
        return V
    
    def null_cond(self, V:Vec, x:np.ndarray) -> np.ndarray:
        """v: A three-vector described in Cartesian coordinates (relative to a stationary observer at infinity).

        x: A spacetime point described in Minkowski coordinates (relative to a stationary observer at infinity).
    
        Returns the normed vector (norm=0) at x in metric coordinates whose spatial components are the transformed components of vec."""

        J = self.jacobian(x); g = self.sample_g(self.coord_pos(x))
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
        disc = b**2 - 4*a*c
        v[0] = min((-b-np.sqrt(disc))/(2*a), (-b+np.sqrt(disc))/(2*a))

        return self.coord_vel(v, x)                    
    

# Render Settings
default_scene = [Hittable(tag=HITTABLE_NULL,
                 shape=Shape(pos=Vec(0,0,0),rot=(0,0,0),tag=SHAPE_NULL))]

spec_settings = [("w", int64), ("h", int64), ("aspect", float64),
                 ("cam_pos", Vec.class_type.instance_type), ("cam_dir", Vec.class_type.instance_type),
                 ("cam_u", Vec.class_type.instance_type), ("cam_v", Vec.class_type.instance_type),
                 ("scene", types.ListType(Hittable.class_type.instance_type)), ("background", float64[:,:,::1]),
                 ("bg_rad", float64), ("grav_field", GravField.class_type.instance_type)]
@jitclass(spec_settings)
class RenderSettings:
    """The settings of the rendered scene. Includes parameters like camera position and angle, scene, background image, etc.
    
    For a scene with no object, simply omit the scene argument, or pass a list of a Hittable with a null Shape.
    
    bg_rad: Radius of the background box."""

    def __init__(self, w:int=800, h:int=600, cam_pos:Vec=Vec(0,0,0), cam_dir:Vec=Vec(1,0,0), # Width, height of image, position and direction of camera
                 scene:list[Hittable]=default_scene, # List of hittables in the scene
                 background:np.ndarray=np.zeros((1,1,3)), bg_rad:float=100, # Background image (default black)
                 grav_field:GravField=GravField(tag=GRAVFIELD_MINKOWSKI)): # Background gravitational field
        self.w = w
        self.h = h
        self.cam_pos = cam_pos
        self.cam_dir = cam_dir
        self.scene = List(scene)
        self.background = background
        self.bg_rad = bg_rad
        self.grav_field = grav_field

        self.aspect = self.w / self.h
        self.cam_u = self.cam_dir.cross(Vec(0,0,1)).normal() # Need to deal with cam_dir being close to Vec(0,0,1)
        self.cam_v = self.cam_u.cross(self.cam_dir).normal()

        for obj in scene:
            if (obj.shape.pos - cam_pos).length() > bg_rad: raise ValueError("All objects must be within the radius of the background image.")

    def ray_dir_px(self, x:int, y:int) -> Vec:
        u = (2*(x+0.5)/self.w - 1) * self.aspect # Viewport coordinates, varies from -aspect to +aspect
        v = 1 - 2*(y+0.5)/self.h # varies from -1 to +1
        ray_dir = (self.cam_dir + (u * self.cam_u) + (v * self.cam_v)).normal()
        return ray_dir
    
    def sample_bg(self, theta:float, phi:float) -> Vec:
        u, v = (phi+np.pi)/(2*np.pi), theta/np.pi
        h, w, _ = self.background.shape
        x, y = int(u*(w-1)), int(v*(h-1)) # Pixel coordinate on the image
        col_array = self.background[y,x]
        return Vec(col_array[0], col_array[1], col_array[2])
    
    def geodesic_eq(self, _, y):
        x, v = y[:4], y[4:]
        chr_syms = self.grav_field.sample_Gamma(x)
        A = np.zeros(4)
        for c in range(4):
            for a in range(4):
                for b in range(4):
                    A[c] -= chr_syms[c,a,b] * v[a] * v[b]
        return np.array([v[0], v[1], v[2], v[3], A[0], A[1], A[2], A[3]])

    def escape(self, Y) -> float: # Event function for when the ray hits the background
        pos_metric = np.array([Y[0], Y[1], Y[2], Y[3]])
        pos_mink = self.grav_field.mink_pos(pos_metric) 
        pt = Vec(pos_mink[1], pos_mink[2], pos_mink[3])
        return (pt - self.cam_pos).length() - self.bg_rad