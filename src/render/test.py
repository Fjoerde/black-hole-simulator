# A place for testing out functions/methods that would not otherwise be directly called.
import numpy as np
from numba_kdtree import KDTree
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

grid = np.load("Classes/GravField/Points/grid_pts.npy")
dist, neighbors, _ = KDTree(grid).query(np.array([0,0,0,0]), k=8)
print(dist.shape, neighbors.shape)
