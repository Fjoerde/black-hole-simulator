from numba import njit

@njit
def init(self, col_field):
    self.col_field = col_field

@njit
def color(self, pt):
    if not self.shape.on_surface(pt): raise ValueError("Expected a point on the surface of the Shape.")
    u, v = self.shape.projection(pt)
    return self.col_field.sample(u, v)