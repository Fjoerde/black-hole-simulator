from numba import njit

@njit
def init(self, col1, col2, n):
    self.n = n
    self.col1 = col1
    self.col2 = col2

@njit
def get_col(self, pt):
    if not self.shape.on_surface(pt): raise ValueError("Expected a point on the surface of the Shape.")
    u, v = self.shape.projection(pt)
    grid_u, grid_v = int(self.n*u), int(self.n*v)
    col_num = (grid_u + grid_v) % 2
    if col_num == 0: return self.col1
    elif col_num == 1: return self.col2