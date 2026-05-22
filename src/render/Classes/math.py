import numpy as np
from numba import float64, int64
from numba.experimental import jitclass
from numba.core import types
from numba.typed import List
from numba.extending import overload
import operator
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Vectors
spec_vec = [("x", float64), ("y", float64), ("z", float64)]
@jitclass(spec_vec)
class Vec:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __add__(self, vec): return Vec(self.x+vec.x, self.y+vec.y, self.z+vec.z)
    def __sub__(self, vec): return Vec(self.x-vec.x, self.y-vec.y, self.z-vec.z)
    def __mul__(self, scal): return Vec(self.x*scal, self.y*scal, self.z*scal)
    def __rmul__(self, scal): return self * scal
    def __truediv__(self, scal): return Vec(self.x/scal, self.y/scal, self.z/scal)
    def __neg__(self): return Vec(-self.x, -self.y, -self.z)
    def __eq__(self, vec): return (self.x == vec.x and self.y == vec.y and self.z == vec.z)
    def __ne__(self, vec): return (self.x != vec.x or self.y != vec.y or self.z != vec.z)

    def length(self): return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    def normal(self):
        len_ = self.length()
        if len_ < 1e-8: return Vec(0,0,0) # Avoid division by zero
        return self / len_
    def dot(self, vec): return (self.x*vec.x + self.y*vec.y + self.z*vec.z)
    def cross(self, vec): return Vec(self.y*vec.z-vec.y*self.z, self.z*vec.x-self.x*vec.z, self.x*vec.y-self.y*vec.x)
    def np_array(self): return np.array([self.x, self.y, self.z], dtype=np.float64) # Turn Vec into numpy array

    def is_normal(self):
        if np.abs(self.length() - 1.) > 1e-6: raise ValueError("Expected normalized vector.")
    def four_vec(self, t): return np.ascontiguousarray(np.array([t, self.x, self.y, self.z], dtype=np.float64))

@overload(operator.mul)
def vec_mul(v1, v2):
    if v1 == Vec.class_type.instance_type and isinstance(v2, Vec.class_type.instance_type): raise ValueError("Encountered multiplication of vectors.")
    elif v1 == float64 and v2 == Vec.class_type.instance_type: return lambda v1,v2: Vec(v2.x*v1, v2.y*v1, v2.z*v1)
    elif v1 == Vec.class_type.instance_type and v2 == float64: return lambda v1,v2: Vec(v1.x*v2, v1.y*v2, v1.z*v2)


# Patches
spec_patch = [("arr", types.ListType(float64[::1])), ("arr_lens", int64[::1]), ("dim", int64),
              ("parent_idx", int64), ("child_idx", types.ListType(int64)), ("pts", float64[:,::1]), ("n_pts", int64),
              ("mins", float64[::1]), ("maxs", float64[::1]), ("corners", float64[:,::1])]
@jitclass(spec_patch)
class Patch:
    """A set of points given as a Cartesian product of 1D-arrays.
    
    arr: A list of 1D-arrays which form the desired set of patch points."""

    def __init__(self, arr:list[np.ndarray]):
        self.arr = List(arr)
        self.dim = len(arr)
        self.parent_idx = -1
        self.child_idx:list[int] = List.empty_list(int64)
        self.arr_lens = self.get_arr_lens()
        self.n_pts = np.prod(self.arr_lens)
        self.pts = self.get_pts()
        self.mins, self.maxs = self.get_extrema()
        self.corners = self.get_corners()

        if len(arr) == 0: raise ValueError("Patch must have at least dimension 1.")
        for i in arr:
            if len(i) == 0: raise ValueError("Dimension of the grid must be at least 1 in each dimension.")

    def get_arr_lens(self) -> np.ndarray[int]:
        """Returns the lengths of the 1D arrays that form the patch."""

        lens = np.zeros(self.dim, dtype=np.int64)
        for i in range(self.dim): lens[i] = len(self.arr[i])
        return lens

    def get_pts(self) -> np.ndarray[float]:
        """Returns the Cartesian product of all 1D arrays in arr."""

        all_pts = np.empty((self.n_pts, self.dim), dtype=np.float64)
        for i in range(self.n_pts):
            idx = i
            for col in range(self.dim-1, -1, -1):
                all_pts[i,col] = self.arr[col][idx % self.arr_lens[col]]
                idx //= self.arr_lens[col]
        return all_pts

    def get_extrema(self) -> tuple[np.ndarray[float]]:
        """Returns the smallest and largest coordinate of the patch along each dimension."""

        mins, maxs = np.empty(self.dim, dtype=np.float64), np.empty(self.dim, dtype=np.float64)
        for i in range(self.dim):
            mins[i] = self.arr[i][0]
            maxs[i] = self.arr[i][-1]
        return mins, maxs

    def get_corners(self) -> np.ndarray[float]:
        """Returns the corners of the patch."""

        corners = np.empty((2**self.dim, self.dim), dtype=np.float64)
        for i in range(2**self.dim):
            for j in range(self.dim):
                if i & 2**j == 0: corners[i,j] = self.mins[j]
                else: corners[i,j] = self.maxs[j]
        return corners
    
    def is_patch_pt(self, pt:np.ndarray) -> bool:
        """Returns if a given pt is a patch point."""

        if pt.shape[0] != self.dim: raise ValueError("pt must have the same dimensions as the patch.")
        for i in range(self.dim):
            if pt[i] not in self.arr[i]: return False
        return True
    
    def in_patch(self, pt:np.ndarray, on_bdary:bool=True) -> bool:
        """Returns if a given pt is inside the patch.
        
        on_bdary: If a point on the boundary of the grid counts as being within the grid."""

        if pt.shape[0] != self.dim: raise ValueError("pt must have the same dimensions as the patch.")
        for i in range(self.dim):
            if not self.mins[i] < pt[i] < self.maxs[i]:
                if not on_bdary: return False
                elif pt[i] != self.mins[i] and pt[i] != self.maxs[i]: return False
        return True

    def on_bdary(self, pt:np.ndarray, pos:bool=True, neg:bool=True) -> bool:
        """Returns if a given pt is on the boundary of the patch.
        
        pos, neg: Specify if the pt can be on the positive/negative boundary."""
        
        on_bdary = False
        for dim in range(self.dim):
            if np.abs(self.arr[dim][-1] - pt[dim]) <= 1e-16:
                if not pos: return False
                on_bdary = True
            if np.abs(self.arr[dim][0] - pt[dim]) <= 1e-16:
                if not neg: return False
                on_bdary = True
        return on_bdary
    
    def adj_pt(self, pts:np.ndarray, dim:int) -> np.ndarray[float]:
        """Returns the patch point adjacent to pt in the direction of dim.
        
        dim: Integer between 1 and dim. The dimension specifying which adjacent point is desired. 
        Negative indicates the adjacent point in the negative direction."""

        if not (1 <= np.abs(dim) <= self.dim): raise ValueError("Invalid value for dim.")

        N = len(pts)
        multi_idx = np.zeros((N, self.dim), dtype=np.int64)
        for i in range(N):
            for j in range(self.dim): multi_idx[i,j] = np.where(self.arr[j] == pts[i,j])[0][0]
        addend = np.zeros((N, self.dim), dtype=np.int64)
        addend[:, np.abs(dim)-1] = np.zeros(N, dtype=np.int64) + np.sign(dim)
        adj_multi_idx = np.clip(multi_idx + addend, 0, (self.arr_lens - 1).repeat(N).reshape(self.dim, N).T)
        new_pts = np.empty((N, self.dim), dtype=np.float64)
        for i in range(N):
            for j in range(self.dim): new_pts[i,j] = self.arr[j][adj_multi_idx[i,j]]
        return new_pts

    def get_idx(self, pts:np.ndarray) -> np.ndarray[int]:
        """Returns the linearized index of a patch point."""

        multi_idx = np.zeros((len(pts), self.dim), dtype=np.int64)
        for i in range(len(pts)):
            for j in range(self.dim):
                entry = np.where(self.arr[j] == pts[i,j])[0]
                if len(entry) == 0: raise ValueError("At least one point in pts is not a grid point.")
                multi_idx[i,j] = entry[0]
        lin_idx = np.zeros(len(pts), dtype=np.int64)
        for i in range(self.dim): lin_idx += multi_idx[:,i] * np.prod(self.arr_lens[i+1:])
        return lin_idx


# Grids
spec_grid = [("patches", types.ListType(Patch.class_type.instance_type)), ("patch_idx", int64[::1]),
             ("n_patch_pts", int64[::1]), ("del_idx", types.ListType(int64[::1])), ("n_del_pts", int64[::1]),
             ("pts", float64[:,::1]), ("dim", int64)]
@jitclass(spec_grid)
class Grid:
    """A grid is a hierarchical structure of Patches. Each patch can inherit child patches whose corners snap to the
    points of the parent. Child patches of the same parent cannot overlap with each other."""

    def __init__(self, patch:Patch=Patch([np.zeros(1, dtype=np.float64)])):
        self.patches:list[Patch] = List([patch])
        self.pts = np.ascontiguousarray(patch.pts)
        self.patch_idx:np.ndarray[int] = np.zeros(len(patch.pts), dtype=np.int64)
        self.n_patch_pts = np.array([patch.n_pts], dtype=np.int64)
        self.del_idx:list[np.ndarray] = List([-np.ones(1, dtype=np.int64)])
        self.n_del_pts = np.zeros(1, dtype=np.int64)
        self.dim = patch.dim

    def finest_patch(self, pts:np.ndarray, min_corner:bool=False) -> np.ndarray[int]:
        """Returns the smallest child among the hierarchies of patches the pt is in. If a pt does not belong to any patch,
        return the root patch.
        
        min_corner: Consider also if there exists a cell within that child in which pt is a minimum
        corner of."""

        if pts.shape[1] != self.dim: raise ValueError("pt must have the same dimensions as the grid.")
        patch_idxs = np.empty(len(pts), dtype=np.int64)
        for i in range(len(pts)):
            pt = pts[i]; idx = 0
            while len(self.patches[idx].child_idx) != 0:
                in_child = False; patch = self.patches[idx]
                for j in patch.child_idx:
                    child = self.patches[j]
                    if child.in_patch(pt):
                        if (not min_corner) or (min_corner and child.on_bdary(pt, pos=False)): 
                            in_child = True; idx = j; break
                if in_child: continue
                else: patch_idxs[i] = idx; break
            patch_idxs[i] = idx
        return patch_idxs

    def get_idx(self, pts:np.ndarray) -> np.ndarray[int]:
        """Returns the indices of an array of grid pts."""

        patch_sort_lst = []; idx_lst = []
        for _ in range(len(self.patches)):
            patch_sort_lst.append(np.empty((1, self.dim), dtype=np.float64))
            idx_lst.append(np.empty(1, dtype=np.int64))
        patch_idxs = self.finest_patch(pts)
        for i in range(len(pts)):
            patch_idx = patch_idxs[i]
            patch_arr = patch_sort_lst[patch_idx]; idx_arr = idx_lst[patch_idx]
            patch_sort_lst[patch_idx] = np.vstack((patch_arr, pts[i].reshape(1, self.dim)))
            idx_lst[patch_idx] = np.append(idx_arr, i)
        indices = np.empty(len(pts), dtype=np.int64)
        for i in range(len(self.patches)):
            if len(patch_sort_lst[i]) == 1: continue
            patch_sort_lst[i] = patch_sort_lst[i][1:] # Remove placeholder
            idx_lst[i] = idx_lst[i][1:] # Remove placeholder
            patch = self.patches[i]
            patch_idxs = patch.get_idx(patch_sort_lst[i])
            patch_idxs += np.sum(self.n_patch_pts[:i] - self.n_del_pts[:i])
            del_pts = self.del_idx[i]
            for j in range(len(patch_idxs)):
                if del_pts[0] == -1: grid_idx = patch_idxs[j]
                else: grid_idx = patch_idxs[j] - len(del_pts[del_pts < patch_idxs[j]])
                indices[idx_lst[i][j]] = grid_idx
        return indices

    def adj_pt(self, pts:np.ndarray, dim:int) -> np.ndarray:
        """Returns the adjacent points of an array of grid points in direction specified by dim."""

        patch_sort_lst = []; idx_lst = []
        for _ in range(len(self.patches)):
            patch_sort_lst.append(np.empty((1, self.dim), dtype=np.float64))
            idx_lst.append(np.empty(1, dtype=np.int64))
        patch_idxs = self.finest_patch(pts)
        for i in range(len(pts)):
            patch_idx = patch_idxs[i]
            patch_arr = patch_sort_lst[patch_idx]; idx_arr = idx_lst[patch_idx]
            patch_sort_lst[patch_idx] = np.vstack((patch_arr, pts[i].reshape(1, self.dim)))
            idx_lst[patch_idx] = np.append(idx_arr, i)
        adj_pts = np.empty((len(pts), self.dim), dtype=np.float64)
        for i in range(len(self.patches)):
            if len(patch_sort_lst[i]) == 1: continue
            patch_sort_lst[i] = patch_sort_lst[i][1:] # Remove placeholder
            idx_lst[i] = idx_lst[i][1:] # Remove placeholder
            patch = self.patches[i]
            adj_pt_patch = patch.adj_pt(patch_sort_lst[i], dim)
            for j in range(len(adj_pt_patch)):
                adj_pts[idx_lst[i][j]] = adj_pt_patch[j]
        return adj_pts

    def add_patch(self, patch:Patch, parent_idx:int=0):
        """Add a patch to the grid.
        
        parent_idx: The index of the parent patch. Default is 0, which indicates the root patch."""

        # Check validity
        parent = self.patches[parent_idx]
        if patch.dim != parent.dim: raise ValueError("The dimensions of the patch must be the same as its parent.")
        patch.parent_idx = parent_idx
        for i in parent.child_idx:
            child = self.patches[i]
            if patch.in_patch(child.mins) and patch.in_patch(child.maxs):
                raise ValueError("The patch must not overlap with any other children of the same parent.")
        for corner in patch.corners:
            if not parent.is_patch_pt(corner): raise ValueError("The corners must snap to the patch points of the parent.")
            for i in parent.child_idx:
                if self.patches[i].in_patch(corner, on_bdary=False):
                    raise ValueError("The patch must not overlap with any other children of the same parent.")
    
        # Remove redundant points
        del_patch_idx = np.empty(1, dtype=np.int64)
        del_pts = np.empty((1, self.dim), dtype=np.float64)
        for i in range(parent.n_pts):
            pt = parent.pts[i]
            if (patch.in_patch(pt, on_bdary=False) or (patch.in_patch(pt) and patch.is_patch_pt(pt))):
                del_patch_idx = np.append(del_patch_idx, i)
                del_pts = np.vstack((del_pts, pt.reshape(1, self.dim)))
        del_patch_idx = del_patch_idx[1:]; del_pts = del_pts[1:] # Remove placeholder
        del_idx = self.get_idx(del_pts)
        mask = np.ones(len(self.pts), dtype=np.bool_)
        for i in range(len(self.pts)):
            if i in del_idx: mask[i] = 0
        self.pts = self.pts[mask]; self.patch_idx = self.patch_idx[mask]
        if self.del_idx[parent_idx][0] == -1: self.del_idx[parent_idx] = del_patch_idx
        else: self.del_idx[parent_idx] = np.concatenate((self.del_idx[parent_idx], del_patch_idx))
        self.n_del_pts[parent_idx] += len(del_patch_idx)

        # Update
        new_patch_idx = np.zeros(len(patch.pts), dtype=np.int64) + len(self.patches)
        self.patch_idx = np.concatenate((self.patch_idx, new_patch_idx))
        self.pts = np.vstack((self.pts, patch.pts))
        self.n_patch_pts = np.append(self.n_patch_pts, patch.n_pts)
        self.del_idx.append(-np.ones(1, dtype=np.int64))
        self.n_del_pts = np.append(self.n_del_pts, 0)
        parent.child_idx.append(len(self.patches))
        self.patches.append(patch)
        return self   
 
    def get_cell(self, pts:np.ndarray) -> tuple[np.ndarray[float]]:
        """Returns the cell in the finest patch the pt is in.
        If pt is outside the grid, return the closest cell.
        If pt is a grid point, return the cell that it is a minimum corner of.
        
        Returns the list of corners, and the minimum and maximum corners."""
        
        min_cell = np.empty((len(pts), self.dim), dtype=np.float64)
        max_cell = np.empty((len(pts), self.dim), dtype=np.float64)
        patch_idxs = self.finest_patch(pts, min_corner=True)
        for i in range(len(pts)):
            patch = self.patches[patch_idxs[i]]
            pt = np.clip(pts[i], patch.mins, patch.maxs) # Assume vanishing partial derivative outside the grid
            for j in range(self.dim):
                idx = np.where(patch.arr[j] <= pt[j])[0][-1]
                min_cell[i,j] = patch.arr[j][idx]
                if idx == len(patch.arr[j])-1: max_cell[i,j] = patch.arr[j][idx] 
                else: max_cell[i,j] = patch.arr[j][idx+1]
        corners = np.empty((len(pts), 2**self.dim, self.dim), dtype=np.float64)
        for i in range(1 << self.dim):
            for j in range(self.dim):
                if i & (1 << j): corners[:,i,j] = max_cell[:,j]
                else: corners[:,i,j] = min_cell[:,j]
        return corners, min_cell, max_cell


# Functions
spec_func = [("grid", Grid.class_type.instance_type), ("dim", int64), ("entries", int64),
             ("vals", float64[:,::1]), ("dvals", float64[:,:,::1])]
@jitclass(spec_func)
class Function:
    """A function whose value is given over a grid. Can be used to perform operations on like how normal functions would be.
    Approximation of the value of the function in between grid points is done using a cardinal cubic Hermite spline. 
    
    grid: The grid the function is given over.
    
    vals: The value of the function at the grid points. Can be an array of numbers.
    A value with some index in vals is assigned to the grid point with the same index in grid.pts."""

    def __init__(self, grid:Grid=Grid(), vals:np.ndarray=np.zeros((1,1), dtype=np.float64)):
        if len(vals.shape) == 1: raise ValueError("Use an array of shape (N, 1) for a function of 1 entry.")
        if len(vals) != len(grid.pts): raise ValueError("vals must have the same number of entries as grid points.")

        self.grid = grid
        self.dim = grid.dim
        self.entries = vals.shape[1]
        self.dvals = np.empty((2**grid.dim, len(grid.pts), self.entries), dtype=np.float64)
        self.vals = self.dvals[0] = vals
        for i in range(1, 2**self.dim):
            bit_len = 0; x = i
            while x: bit_len += 1; x >>= 1
            dval = self.dvals[i - (1 << (bit_len-1))]
            dval = self.diff_vals(dval, bit_len-1)
            self.dvals[i] = dval
    
    def diff_vals(self, vals:np.ndarray, dim:int) -> np.ndarray[float]:
        """Returns the derivative of the function over the grid given by vals with respect to the dimension.
        
        vals: The value corresponding to every point within the grid."""
        
        if vals.shape[0] != len(self.grid.pts): raise ValueError("Number of values not equal to the number of grid points.")
        if not (0 <= dim <= self.dim-1): raise ValueError("Invalid value for dim.")

        N = len(self.grid.pts)
        adj_p = self.grid.adj_pt(self.grid.pts, dim+1)
        adj_m = self.grid.adj_pt(self.grid.pts, -(dim+1))
        indices = self.grid.get_idx(np.vstack((adj_p, adj_m)))
        idx_p = indices[:N]; idx_m = indices[N:]
        h = (adj_p - adj_m)[:,dim].repeat(self.entries).reshape(N, self.entries)
        delta = vals[idx_p] - vals[idx_m]
        derivs = np.where(h > 0, delta / h, 0)
        return derivs

    def interp(self, pts:np.ndarray) -> np.ndarray[float]:
        """Returns the interpolated values of the function at pts.
        
        Returns: Shape (len(pts), self.entries), the interpolated value of the function at each point."""

        n = 2 ** self.dim
        corners, min_cell, max_cell = self.grid.get_cell(pts)
        corner_idx = self.grid.get_idx(corners.reshape(len(pts)*n, self.dim)).reshape(len(pts), n)
        dval = np.empty((len(pts), n, n, self.entries), dtype=np.float64) # 2nd index is corner, 3rd index is derivative
        h = (max_cell - min_cell).repeat(n).reshape(len(pts), self.dim, n)
        for i in range(len(pts)):
            for j in range(n): dval[i,j] = self.dvals[:, corner_idx[i,j]]
        for deriv in range(n): # For chain rule
            for dim in range(self.dim):
                if not deriv & (1 << dim): h[:,dim,deriv] = 1
        for i in range(len(pts)):
            for j in range(n): dval[i,:,j] *= np.prod(h[i,:,j]) # Chain rule
        normed = np.clip(np.where(max_cell-min_cell > 1e-16, (pts-min_cell)/(max_cell-min_cell), 0), 0, 1) # Normalized coords
        for i in range(self.dim-1, -1, -1):
            for corner in range(1 << i): # For every pair of corners
                f0 = dval[:, corner, :(1<<i)]; f1 = dval[:, corner+(1<<i), :(1<<i)]
                df0 = dval[:, corner, (1<<i):(1<<(i+1))]; df1 = dval[:, corner+(1<<i), (1<<i):(1<<(i+1))]
                u = np.repeat(normed[:,i], (1<<i)*self.entries).reshape(len(pts), 1<<i, self.entries)
                # Interpolate
                h11 = 2*f0 - 2*f1 + df0 + df1
                h10 = -3*f0 + 3*f1 - 2*df0 - df1
                h01 = df0
                h00 = f0
                dval[:, corner, :(1<<i)] = h00 + u*(h01 + u*(h10 + u*h11)) # Horner's method
        return np.ascontiguousarray(dval[:,0,0])

    def integrate_cell(self, corners:np.ndarray, min_corners:np.ndarray, max_corners:np.ndarray,
                       norm_mins:np.ndarray, norm_maxs:np.ndarray) -> np.ndarray[float]:
        """Returns the integral of the function over different cells, uniquely specified by its minimum corner.
        
        corners: The corners of the cells integrated over. No check is done to ensure that they are grid points.

        min_corners, max_corners: The minimum and maximum corners of the cells integrated over.
        No check is done to ensure that they are consistent with corners (variable).
        
        norm_mins, norm_maxs: The normalized minimum and maximum coordiantes of the integration region within the cell."""

        if not (corners.shape[0] == min_corners.shape[0] == max_corners.shape[0] == norm_mins.shape[0] == norm_maxs.shape[0]):
            raise ValueError("pts, norm_mins, and norm_maxs must have the same length.")
        if not (corners.shape[2] == min_corners.shape[1] == max_corners.shape[1] == norm_mins.shape[1] == norm_maxs.shape[1] == self.dim):
            raise ValueError("pts, norm_mins, and norm_maxs must have the same dimension as grid.")
        if not (np.all((norm_mins >= 0) & (norm_mins <= 1)) and np.all((norm_maxs >= 0) & (norm_maxs <= 1))): raise ValueError("norm_min and norm_max are not normalized.")
        if not np.all(norm_mins <= norm_maxs): raise ValueError("norm_mins and norm_maxs are not the minimum/maximum corners of the integration region within the cell.")
        if not np.all(min_corners <= max_corners): raise ValueError("min_corners and max_corners are not the minimum/maximum corners of the cell integrated over.")
        
        n = 2 ** self.dim; N = len(corners)
        h = max_corners - min_corners
        for i in range(N):
            if np.min(np.abs(max_corners[i] - min_corners[i])) <= 1e-16: raise ValueError("None of the pts can be on the positive boundary of the root patch.")
        corner_idx = self.grid.get_idx(corners.reshape(N*n, self.dim)).reshape(N, n)
        dval = np.empty((N, n, n, self.entries), dtype=np.float64) # 2nd index is derivative, 3rd is corner
        for i in range(N):
            for j in range(n): dval[i,:,j] = self.dvals[:, corner_idx[i,j]] # Collect derivative values on corners
        bases = np.empty((N, self.dim, 2, 2), dtype=np.float64)
        for dim in range(self.dim): # Integral of Hermite basis functions
            t_min = norm_mins[:,dim]; t_max = norm_maxs[:,dim]
            bases[:,dim,0,0] = (t_max**4/2 - t_max**3 + t_max) - (t_min**4/2 - t_min**3 + t_min)
            bases[:,dim,0,1] = (-t_max**4/2 + t_max**3) - (-t_min**4/2 + t_min**3)
            bases[:,dim,1,0] = (t_max**4/4 - 2*t_max**3/3 + t_max**2/2) - (t_min**4/4 - 2*t_min**3/3 + t_min**2/2)
            bases[:,dim,1,1] = (t_max**4/4 - t_max**3/3) - (t_min**4/4 - t_min**3/3)
            bases[:,dim] *= h[:,dim].repeat(4).reshape(N, 2, 2)
        total = np.zeros((N, self.entries), dtype=np.float64)
        for i in range(n): # For each derivative
            s = np.zeros((N, self.entries), dtype=np.float64)
            for j in range(n): # For each corner
                term = dval[:,i,j]
                for k in range(self.dim):
                    p = min(i & (1 << k), 1); q = min(j & (1 << k), 1)
                    term *= bases[:,k,p,q].repeat(self.entries).reshape(N, self.entries)
                s += term
            for j in range(self.dim): # Multiply by appropriate lengths due to chain rule
                if i & (1 << j): s *= h[:,j].repeat(self.entries).reshape(N, self.entries)
            total += s
        return total

    def integrate(self, min:np.ndarray, max:np.ndarray) -> np.ndarray[float]:
        """Return the integral of the function over an axis-aligned box defined by the minimum and maximum corners."""
        
        if min.shape[0] != self.dim or max.shape[0] != self.dim: raise ValueError("min and max must have the same dimension as the grid.")
        
        corners, min_cell, max_cell = self.grid.get_cell(self.grid.pts)
        h = max_cell - min_cell
        keep = np.zeros(len(self.grid.pts), dtype=np.bool_)
        normed = np.zeros((len(self.grid.pts), 2, self.dim), dtype=np.float64)
        for i in range(len(corners)):
            if np.max(np.abs(h[i])) <= 1e-16: continue # Reject if defined cell is degenerate
            in_box = False
            for j in range(2**self.dim):
                if np.all((corners[i,j] >= min) & (corners[i,j] <= max)): in_box = True; keep[i] = 1; break
            if in_box:
                normed[i,0] = np.clip((min - min_cell[i]) / h[i], 0, 1)
                normed[i,1] = np.clip((max - min_cell[i]) / h[i], 0, 1)
        corners = corners[keep]; min_cell = min_cell[keep]; max_cell = max_cell[keep]; normed = normed[keep]
        norm_mins = normed[:,0]; norm_maxs = normed[:,1]
        int_vals = self.integrate_cell(corners, min_cell, max_cell, norm_mins, norm_maxs)
        result = np.empty(self.entries, dtype=np.float64)
        for i in range(self.entries): result[i] = np.sum(int_vals[:,i])
        return np.ascontiguousarray(result)


# Color Converters
spec_colconvert = [("grid", Grid.class_type.instance_type), ("sorted_wvls", float64[::1]), ("A_W", float64[:,::1]),
                   ("D", float64[:,::1]), ("norm_scal", float64), ("norm_lux", float64)]
@jitclass(spec_colconvert)
class ColConverter:
    """Calculate the spectral intensity (in W m^-2 nm^-1) and RGB values for a given color.
    Call get_spec_int() to convert an RGB color to a spectral intensity and vice versa for get_rgb().
    The CIE 1931 colorimetric observer is used to convert a spectral intensity to an RGB color.
    To convert an RGB color to a spectral intensity, the one with the least hyperbolic tangent squared slope (LHTSS) is picked.
    
    norm_lux: A reference luminous intensity, equal to that of bright daylight (blackbody spectrum at 6504 K).
    Every conversion scales accordingly (so bright white gives back the blackbody spectrum)."""

    def __init__(self, norm_lux:float=1e5):
        self.norm_lux = norm_lux
        self.grid = Grid(Patch([np.linspace(380, 780, 41)]))
        self.sorted_wvls = np.sort(np.ascontiguousarray(self.grid.pts).reshape(len(self.grid.pts)))
        self.norm_scal = self.get_norm_scal()
        self.A_W = self.get_A_W()
        self.D = self.get_D()
    
    def gauss(self, wvls:np.ndarray, m:float, t1:float, t2:float) -> np.ndarray:
        """The piecewise gaussians for the color-matching functions."""

        val1 = np.exp(-t1**2 * (wvls-m)**2 / 2)
        val2 = np.exp(-t2**2 * (wvls-m)**2 / 2)
        return np.where(wvls < m, val1, val2)

    def get_col_bases(self, wvls:np.ndarray) -> np.ndarray:
        """Returns the color-matching functions at the inputted wavelengths."""

        x = 1.056*self.gauss(wvls, 599.8, 0.0264, 0.0323) + 0.362*self.gauss(wvls, 442, 0.0624, 0.0374) - 0.065*self.gauss(wvls, 501.1, 0.049, 0.0382)
        y = 0.821*self.gauss(wvls, 568.8, 0.0213, 0.0247) + 0.286*self.gauss(wvls, 530.9, 0.0613, 0.0322)
        z = 1.217*self.gauss(wvls, 437, 0.0845, 0.0278) + 0.681*self.gauss(wvls, 459, 0.0385, 0.0725)
        return np.vstack((x, y, z))
    
    def d65(self, wvls:np.ndarray) -> np.ndarray:
        """Returns the spectral value of the blackbody spectrum at 6504 K. Acts as an illuminant."""

        h = 6.62607015e-34
        c = 2.99792458e8
        k = 1.380649e-23
        T = 6504; wvls_si = wvls * 1e-9
        return (2*h*c**2/wvls_si**5) / (np.exp(h*c/(wvls_si*k*T))-1) * 1e-9

    def get_norm_scal(self) -> float:
        """Returns the normalizing scalar so that bright white (the D65 spectrum) returns Y=100."""

        wvls = self.grid.pts.reshape(len(self.grid.pts))
        coly = self.get_col_bases(wvls)[1]
        lb, ub = np.array([380], dtype=np.float64), np.array([780], dtype=np.float64)
        vals = self.d65(wvls) * coly
        unscal_y = Function(self.grid, vals.reshape(len(vals), 1)).integrate(lb, ub)[0]
        return 100 / unscal_y

    def get_A_W(self) -> np.ndarray:
        """The matrix used for the constraint equation on reflectance."""

        wvls = self.sorted_wvls
        col_bases = self.get_col_bases(wvls).T
        W = self.d65(wvls).repeat(3).reshape((len(wvls), 3)) * self.norm_scal
        diff_wvls = np.diff(wvls); diff_wvls = np.append(diff_wvls, diff_wvls[-1])
        diff_wvls = diff_wvls.repeat(3).reshape((len(wvls), 3))
        A_W = col_bases * W * diff_wvls / 100
        return np.ascontiguousarray(A_W)

    def get_D(self) -> np.ndarray:
        """The difference matrix that penalizes big changes in slopes."""

        N = len(self.grid.pts)
        D = np.zeros((N, N), dtype=np.float64)
        h = np.diff(self.sorted_wvls)
        for i in range(N-1):
            inv_h = 1 / h[i]
            D[i,i] += inv_h; D[i+1,i+1] += inv_h
            D[i,i+1] -= inv_h; D[i+1,i] -= inv_h
        return np.ascontiguousarray(D)

    def lhtss_iter(self, z:np.ndarray, tst_vals:np.ndarray) -> np.ndarray: # Newton's method for solving a non-linear system of eqs
        """An iteration of the LHTSS method."""
        
        N = len(self.grid.pts)
        R = (np.tanh(z) + 1) / 2
        dR = np.diag((1 - np.tanh(z)**2) / 2)
        tst_vals_cur = self.A_W.T @ R
        F = np.concatenate((self.D @ z, tst_vals_cur - tst_vals))
        J = np.zeros((N+3, N+3), dtype=np.float64)
        J[:N,:N] = self.D; J[N:,:N] = self.A_W.T @ dR; J[:N,N:] = self.A_W
        delta = np.linalg.inv(J) @ -F
        return delta[:N]
    
    def reflectance(self, tst_vals:np.ndarray) -> np.ndarray:
        """Calculates the reflectance for a given triplet of tristimulus values by solving for the
        least-hyperbolic-tangent-squared-slope (LHTSS) graph (so that it stays between 0 and 1)."""

        z = np.zeros(len(self.grid.pts), dtype=np.float64) # Initial guess, R = 0.5 everywhere
        last_dz = np.inf # To check if sequence is converging
        for _ in range(50):
            dz = self.lhtss_iter(z, tst_vals); z += dz
            max_dz = np.max(np.abs(dz))
            if max_dz <= 1e-4 or max_dz >= last_dz: break # End loop if change is small or if diverging
            last_dz = max_dz
        return (np.tanh(z) + 1) / 2

    def get_tst_vals(self, spec_int:Function) -> np.ndarray:
        """Computes the tristimulus values for a given spectral intensity function."""

        if spec_int.entries != 1 or spec_int.dim != 1: raise ValueError("Invalid spectral intensity.")

        wvls = spec_int.grid.pts.reshape(len(spec_int.grid.pts))
        spec_vals = spec_int.vals.reshape(len(spec_int.grid.pts))
        col_bases = self.get_col_bases(wvls).T
        spec_vals = spec_vals.repeat(3).reshape(len(spec_vals), 3) * col_bases
        lb, ub = np.array([380], dtype=np.float64), np.array([780], dtype=np.float64)
        return Function(spec_int.grid, spec_vals).integrate(lb, ub) * self.norm_scal / 100

    def get_spec_int(self, rgb:np.ndarray, lux:float=-1.) -> Function:
        """The method to call to convert an RGB color into a spectral intensity.
        
        lux: The luminous intensity of the resulting intensity graph. Set to -1 to set it equal to the
        reference value."""

        if rgb.shape[0] != 3: raise ValueError("Invalid shape for rgb.")
        if lux < 0: lux = self.norm_lux

        rgb_lin = np.clip(np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4), 0, 1)
        mat = np.ascontiguousarray(np.array([[0.4124, 0.3576, 0.1805],
                                             [0.2126, 0.7152, 0.0722],
                                             [0.0193, 0.1192, 0.9505]], dtype=np.float64))
        tst_vals = mat @ rgb_lin # In the range 0 to 1
        spec_vals = self.d65(self.sorted_wvls) * self.reflectance(tst_vals) * (lux / self.norm_lux)
        # Sort spec_vals back in order of grid.pts
        new_spec_vals = np.empty((len(self.grid.pts), 1), dtype=np.float64)
        for i in range(len(self.grid.pts)):
            idx = np.where(self.sorted_wvls == self.grid.pts[i,0])[0][0]
            new_spec_vals[i,0] = spec_vals[idx]
        return Function(self.grid, new_spec_vals)
    
    def get_rgb(self, spec_int:Function) -> np.ndarray:
        """The method to call to convert a spectral intensity to an RGB color."""

        tst_vals = self.get_tst_vals(spec_int)
        mat = np.ascontiguousarray(np.array([[3.2406, -1.5372, -0.4986],
                                             [-0.9689, 1.8758, 0.0415],
                                             [0.0557, -0.2040, 1.0570]], dtype=np.float64))
        rgb_lin = mat @ tst_vals
        rgb = np.clip(np.where(rgb_lin <= 0.0031308, rgb_lin*12.92, 1.055*(rgb_lin)**(1/2.4) - 0.055), 0, 1)
        return rgb

def_cc = ColConverter() # Initialize default color converter