from numba import njit

@njit
def init(self, col1, col2, n, col_converter):
    self.n = n
    self.spec_int1 = col_converter.get_spec_int(col1)
    self.spec_int2 = col_converter.get_spec_int(col2)

@njit
def spec_int(self, pt):
    if not self.shape.on_surface(pt): raise ValueError("Expected a point on the surface of the Shape.")
    u, v = self.shape.projection(pt)
    grid_u, grid_v = int(self.n*u), int(self.n*v)
    col_num = (grid_u + grid_v) % 2
    if col_num == 0: return self.spec_int1
    elif col_num == 1: return self.spec_int2