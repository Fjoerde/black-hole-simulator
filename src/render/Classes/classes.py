import numpy as np
from numba import float64, int64
from numba.experimental import jitclass
from numba.core import types
from numba.typed import List

from Classes.base import *
from Classes.tags import *
import Classes.GravField.methods as GravMethod

import Classes.Shapes.null as ShapeNull
import Classes.Shapes.sphere as Sphere
import Classes.Shapes.cylinder as Cylinder
import Classes.Shapes.annulus as Annulus

import Classes.ColFields.const_col as ConstCol
import Classes.ColFields.checkerboard as Checkerboard

import Classes.Hittables.null as HittableNull
import Classes.Hittables.light as Light


# Class
spec_shape = [# For all Shapes
              ("pos", Vec.class_type.instance_type), ("rot", types.Tuple((float64, float64, float64))),
              ("X", Vec.class_type.instance_type), ("Y", Vec.class_type.instance_type), ("Z", Vec.class_type.instance_type),
              ("ray_pos", Vec.class_type.instance_type), ("ray_dir", Vec.class_type.instance_type), ("pt", Vec.class_type.instance_type),
              ("bb", BoundingBox.class_type.instance_type), ("tag", int64), ("y", float64[:]),
              
              # For Sphere and Cylinder
              ("radius", float64),

              # For Cylinder and Annulus
              ("height", float64), ("c_top", Vec.class_type.instance_type), ("c_base", Vec.class_type.instance_type),

              # For Annulus
              ("r_in", float64), ("r_out", float64)]
@jitclass(spec_shape)
class Shape: # Abstract class of any arbitrary geometric shape, i.e. any subclass of Shape must contain the following methods
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

    def in_shape_int(self, y:list) -> float:
        """Returns the distance to the closest point on the shape. Negative if the point is in the shape.

           Used as an event function for numerical integration."""
        
        pos_metric = np.array([y[0], y[1], y[2], y[3]])
        pos_mink = GravMethod.mink_pos(pos_metric) 
        pt = Vec(pos_mink[1], pos_mink[2], pos_mink[3])
        close_pt = self.closest_pt(pt)
        dist = (pt - close_pt).length()
        return np.float64(-dist) if self.in_shape(pt) else np.float64(dist)

    def projection(self, pt:Vec) -> tuple[float, float]:
        """Projects the surface of the shape onto a plane.

           Takes in a point on the surface of the shape and outputs its corresponding coordinate in the domain [0, 1]x[0, 1]."""
        
        if not self.on_surface(pt): raise ValueError("Expected a point on the surface of the shape.")

        if self.tag == SHAPE_NULL: return ShapeNull.projection(self, pt)
        if self.tag == SHAPE_SPHERE: return Sphere.projection(self, pt)
        if self.tag == SHAPE_CYLINDER: return Cylinder.projection(self, pt)
        if self.tag == SHAPE_ANNULUS: return Annulus.projection(self, pt)

    def straight_hit(self, ray_pos, ray_dir) -> float:
        """Detects the parameter along the light ray path at which it would hit the shape assuming it travels in a straight line."""

        if self.tag == SHAPE_NULL: return ShapeNull.straight_hit(self, ray_pos, ray_dir)
        if self.tag == SHAPE_SPHERE: return Sphere.straight_hit(self, ray_pos, ray_dir)
        if self.tag == SHAPE_CYLINDER: return Cylinder.straight_hit(self, ray_pos, ray_dir)
        if self.tag == SHAPE_ANNULUS: return Annulus.straight_hit(self, ray_pos, ray_dir)
    

# Color Fields
spec_colfield = [# For any ColField
                 ("tag", int64),

                 # For ConstCol
                 ("col", Vec.class_type.instance_type),
                 
                 # For Checkerboard
                 ("col1", Vec.class_type.instance_type), ("col2", Vec.class_type.instance_type)]
@jitclass(spec_colfield)
class ColField: # Defined on an abstract uv-plane [0,1]^2 that maps onto the surface of the shape
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
                 ("shape", Shape.class_type.instance_type), ("tag", int64), ("pt", Vec.class_type.instance_type),

                 # For Light
                 ("col_field", ColField.class_type.instance_type)]
@jitclass(spec_hittable)
class Hittable: # Abstract class of hittable objects in a scene
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


# Render Settings
default_scene = [Hittable(tag=HITTABLE_NULL,
                 shape=Shape(pos=Vec(0,0,0),rot=(0,0,0),tag=SHAPE_NULL))]

spec_settings = [("w", int64), ("h", int64), ("aspect", float64), ("x", int64), ("y", int64), ("theta", float64), ("phi", float64),
                 ("pos", float64[:]), ("dir", float64[:]),
                 ("cam_pos", Vec.class_type.instance_type), ("cam_dir", Vec.class_type.instance_type),
                 ("scene", types.ListType(Hittable.class_type.instance_type)), ("background", float64[:,:,::1]),
                 ("bg_rad", float64), ("threshold", float64), ("obj", Hittable.class_type.instance_type),
                 ("cam_u", Vec.class_type.instance_type), ("cam_v", Vec.class_type.instance_type),
                 ("Y", float64[:])]
@jitclass(spec_settings)
class RenderSettings:
    """The settings of the rendered scene. Includes parameters like camera position and angle, scene, background image, etc.
    
    For a scene with no object, simply omit the scene argument, or pass a list of a Hittable with a null Shape.
    
    bg_rad: Radius of the background box.
    
    threshold: Used to determine if a straight-line approximation for a light ray is appropriate. If the Kretschmann scalar along the light ray
    exceeds the threshold, then the approximation is deemed inaccurate."""

    def __init__(self, w:int=800, h:int=600, cam_pos:Vec=Vec(0,0,0), cam_dir:Vec=Vec(1,0,0), # Width, height of image, position and direction of camera
                 scene:list[Hittable]=default_scene, # List of hittables in the scene
                 background:np.ndarray=np.zeros((1,1,3)), bg_rad:float=100, # Background image (default black)
                 threshold:float=1e-6): # Threshold for the Kretschmann scalar to determine if a straight-line approximation is suitable
        self.w = w
        self.h = h
        self.cam_pos = cam_pos
        self.cam_dir = cam_dir
        self.scene = List(scene)
        self.background = background
        self.bg_rad = bg_rad
        self.threshold = threshold

        self.aspect = self.w / self.h
        self.cam_u = self.cam_dir.cross(Vec(0,0,1)).normal() # Need to deal with cam_dir being close to Vec(0,0,1)
        self.cam_v = self.cam_u.cross(self.cam_dir).normal()

        for obj in scene:
            if (obj.shape.pos - cam_pos).length() > bg_rad: raise ValueError("All objects must be within the radius of the background image.")
        if self.threshold <= 0: raise ValueError("Threshold must be a positive number.")

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

    def escape(self, Y) -> float: # Event function for when the ray hits the background
        pos_metric = np.array([Y[0], Y[1], Y[2], Y[3]])
        pos_mink = GravMethod.mink_pos(pos_metric) 
        pt = Vec(pos_mink[1], pos_mink[2], pos_mink[3])
        return (pt - self.cam_pos).length() - self.bg_rad
    
