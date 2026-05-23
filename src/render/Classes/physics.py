import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

from Classes.math import *
from Classes.tags import *

import Classes.Shapes.null as ShapeNull
import Classes.Shapes.sphere as Sphere
import Classes.Shapes.cylinder as Cylinder
import Classes.Shapes.annulus as Annulus

import Classes.Hittables.null as HittableNull
import Classes.Hittables.blackbody as Blackbody
import Classes.Hittables.checkerboard as Checkerboard

import Classes.GravField.minkowski as Minkowski
import Classes.GravField.schwarzschild as Schwarzschild
import Classes.GravField.kerr as Kerr
import Classes.GravField.kerr_newman as KerrNewman

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
                 ("temp", float64), ("specint_grid", Grid.class_type.instance_type),
                 
                 # For Checkerboard
                 ("spec_int1", Function.class_type.instance_type), ("spec_int2", Function.class_type.instance_type),
                 ("n", int64)]
@jitclass(spec_hittable)
class Hittable:
    def __init__(self, shape:Shape, tag:int,
                 temp:float=0.,
                 col1:np.ndarray=np.zeros(3, dtype=np.float64), col2:np.ndarray=np.zeros(3, dtype=np.float64), n:int=10,
                 col_converter:ColConverter=def_cc):
        self.shape = shape
        self.tag = tag

        if self.tag == HITTABLE_NULL: HittableNull.init(self)
        if self.tag == HITTABLE_BLACKBODY: Blackbody.init(self, temp, col_converter)
        if self.tag == HITTABLE_CHECKERBOARD: Checkerboard.init(self, col1, col2, n, col_converter)

    def spec_int(self, pt) -> Function:
        """Returns the spectral intensity emitted by the Hittable at a pt."""

        if not self.shape.on_surface(pt): raise ValueError("Expected a point on the surface of the shape.")

        if self.tag == HITTABLE_NULL: return HittableNull.spec_int(self, pt)
        if self.tag == HITTABLE_BLACKBODY: return Blackbody.spec_int(self, pt)
        if self.tag == HITTABLE_CHECKERBOARD: return Checkerboard.spec_int(self, pt)



# Gravitational Fields
spec_gravfield = [# For all GravFields
                  ("tag", int64),

                  # For Schwarzschild, Kerr, and Kerr-Newman
                  ("pos", Vec.class_type.instance_type), ("M", float64),

                  # For Kerr and Kerr-Newman
                  ("rot", float64[:,::1]), ("rot_inv", float64[:,::1]), ("J", float64),
                  
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

        return self.jacobian(x) @ np.ascontiguousarray(v)
    
    def mink_vel(self, v:np.ndarray, x:np.ndarray) -> np.ndarray:
        """v: A contravariant four-vector described in metric coordinates (relative to a stationary observer at infinity).

        x: A spacetime point described in metric coordinates (relative to a stationary observer at infinity).

        Performs a contravariant transformation of v at x from metric coordinates to Minkowski coordinates."""

        return self.jacobian_inv(x) @ np.ascontiguousarray(v) 

    def timelike_cond(self, V:Vec, x:np.ndarray) -> np.ndarray:
        """v: A three-vector described in Cartesian coordinates (relative to a stationary observer at infinity).

        x: A spacetime point described in Minkowski coordinates (relative to a stationary observer at infinity).
    
        Returns the timelike vector (norm=-1) at x in metric coordinates whose spatial components are the transformed components of vec."""

        J = self.jacobian(x); g = self.sample_g(self.coord_pos(x))
        k = J.T @ g @ J
        v = np.ascontiguousarray(np.array([0, V.x, V.y, V.z]))
        a = k[0,0]; b = 2 * (k[0,1:] @ v[1:]); c = (k[1:,1:] @ v[1:]) @ v[1:] + 1
        disc = b**2 - 4*a*c
        v[0] = max((-b-np.sqrt(disc))/(2*a), (-b+np.sqrt(disc))/(2*a))
        return self.coord_vel(v, x)    
    
    def null_cond(self, V:Vec, x:np.ndarray) -> np.ndarray:
        """v: A three-vector described in Cartesian coordinates (relative to a stationary observer at infinity).

        x: A spacetime point described in Minkowski coordinates (relative to a stationary observer at infinity).
    
        Returns the normed vector (norm=0) at x in metric coordinates whose spatial components are the transformed components of vec."""

        J = self.jacobian(x); g = self.sample_g(self.coord_pos(x))
        k = J.T @ g @ J
        v = np.ascontiguousarray(np.array([0, V.x, V.y, V.z]))
        a = k[0,0]; b = 2 * (k[0,1:] @ v[1:]); c = (k[1:,1:] @ v[1:]) @ v[1:]
        disc = b**2 - 4*a*c
        v[0] = min((-b-np.sqrt(disc))/(2*a), (-b+np.sqrt(disc))/(2*a))
        return self.coord_vel(v, x)


# Gases
null_4d_grid = Grid(Patch([np.zeros(1, dtype=np.float64),
                           np.zeros(1, dtype=np.float64),
                           np.zeros(1, dtype=np.float64),
                           np.zeros(1, dtype=np.float64)]))
def_gas = Function(null_4d_grid, np.array([[1, 0, 0, 0, 0, 0]], dtype=np.float64))