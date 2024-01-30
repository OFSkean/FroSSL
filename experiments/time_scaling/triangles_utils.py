import numpy as np

def permute_point(p, permutation=None):
    """
    Permutes the point according to the permutation keyword argument. The
    default permutation is "012" which does not change the order of the
    coordinate. To rotate counterclockwise, use "120" and to rotate clockwise
    use "201"."""
    if not permutation:
        return p
    return [p[int(permutation[i])] for i in range(len(p))]

def unzip(l):
    """[(a1, b1), ..., (an, bn)] ----> ([a1, ..., an], [b1, ..., bn])"""
    return list(zip(*l))

def project_point(p, permutation=None):
    """
    Maps (x,y,z) coordinates to planar simplex.
    Parameters
    ----------
    p: 3-tuple
    The point to be projected p = (x, y, z)
    permutation: string, None, equivalent to "012"
    The order of the coordinates, counterclockwise from the origin
    """
    permuted = permute_point(p, permutation=permutation)
    a = permuted[0]
    b = permuted[1]
    x = a + b/2.
    y = (np.sqrt(3)/2) * b
    return np.array([x, y])

def fill_region(ax, color, points, pattern=None, zorder=-1000, alpha=None):
    """Draws a triangle behind the plot to serve as the background color
    for a given region."""
    vertices = map(project_point, points)
    xs, ys = unzip(vertices)
    poly = ax.fill(xs, ys, facecolor=color, edgecolor=color, hatch=pattern, zorder=zorder, alpha=alpha)
    return poly
