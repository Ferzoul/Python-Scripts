# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 2020

@author: Fernando Esquivel

Based on "Fast and Accurate Computation of Polyhedral
Mass Properties" algorithm from Brian Mirtich (1996)

Input: an array or list
Output: [Cx, Cy, Cz, m, Ixx, Iyy, Izz, Iyx, Izx, Izy]
Cx, Cy, Cz == Centroids in x-, y- and z-axis.
m == mass, considering the density as 1.
Ixx, Iyy, Izz == Moments of inertia.
Iyx, Izx, Izy == Products of inertia.
"""
from scipy.spatial import ConvexHull
import numpy as np


def AddTriangleContribution(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    # Signed volume of the tetrahedron. Volume should always be positive.
    # If not, the vertexes are in clockwise order and should be reversed.
    v = x1*y2*z3 + y1*z2*x3 + x2*y3*z1 - (x3*y2*z1 + x2*y1*z3 + y3*z2*x1)
    if v < 0:
        [x1, y1, z1, x3, y3, z3] = [x3, y3, z3, x1, y1, z1]
        v = x1*y2*z3 + y1*z2*x3 + x2*y3*z1 - (x3*y2*z1 + x2*y1*z3 + y3*z2*x1)
    # Contribution to the mass
    m = v
    # Contribution to the centroid
    x4 = x1 + x2 + x3
    y4 = y1 + y2 + y3
    z4 = z1 + z2 + z3
    Cx = (v*x4)
    Cy = (v*y4)
    Cz = (v*z4)
    # Contribution to moment of inertia monomials
    xx = v * (x1*x1 + x2*x2 + x3*x3 + x4*x4)
    yy = v * (y1*y1 + y2*y2 + y3*y3 + y4*y4)
    zz = v * (z1*z1 + z2*z2 + z3*z3 + z4*z4)
    yx = v * (y1*x1 + y2*x2 + y3*x3 + y4*x4)
    zx = v * (z1*x1 + z2*x2 + z3*x3 + z4*x4)
    zy = v * (z1*y1 + z2*y2 + z3*y3 + z4*y4)
    return [m, Cx, Cy, Cz, xx, yy, zz, yx, zx, zy]


def GetResults(parameters):
    [m, Cx, Cy, Cz, xx, yy, zz, yx, zx, zy] = parameters
    # Centroid
    r = 1.0 / (4 * m)
    Cx *= r
    Cy *= r
    Cz *= r
    # Mass
    m /= 6
    # Moment of inertia about the centroid.
    r = 1.0 / 120
    Iyx = yx * r - m * Cy*Cx
    Izx = zx * r - m * Cz*Cx
    Izy = zy * r - m * Cz*Cy
    xx = xx * r - m * Cx*Cx
    yy = yy * r - m * Cy*Cy
    zz = zz * r - m * Cz*Cz
    Ixx = yy + zz
    Iyy = zz + xx
    Izz = xx + yy
    return [Cx, Cy, Cz, m, Ixx, Iyy, Izz, Iyx, Izx, Izy]


def TriangleCoords(points):
    index = ConvexHull(points)
    simplex = index.simplices
    shape = np.shape(simplex)
    coords = np.zeros((shape[0], shape[1], 3))
    for i in range(shape[0]):
        for j in range(shape[1]):
            temp = simplex[i][j]
            coords[i][j] = points[temp]
    return coords


def CalculateCentroids(points):
    coords = TriangleCoords(points)
    # initialize parameters for triangle contribution
    parameters = np.zeros(10)
    # Add Triangle Contributions
    for i in range(len(coords)):
        x1 = coords[i][0][0]
        y1 = coords[i][0][1]
        z1 = coords[i][0][2]
        x2 = coords[i][1][0]
        y2 = coords[i][1][1]
        z2 = coords[i][1][2]
        x3 = coords[i][2][0]
        y3 = coords[i][2][1]
        z3 = coords[i][2][2]
        parameters += AddTriangleContribution(x1, y1, z1,
                                              x2, y2, z2,
                                              x3, y3, z3)
    results = GetResults(parameters)
    return results


if __name__ == '__main__':
    # These are some examples of simple geometries to
    # test the script
    test_poly1 = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [1, 1.5, 0],
                           [0, 1.5, 0],
                           [0, 0, 5],
                           [1, 0, 5],
                           [1, 1.5, 3],
                           [0, 1.5, 3],
                           [0, 0.75, 5],
                           [1, 0.75, 5]])
    test_poly2 = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [1, 0, 0],
                           [1, 1, 0],
                           [0.5, 0.5, 1]])
    test_poly3 = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0],
                           [0.5, 0.5, 1],
                           [0.5, 0.5, -1]])
    test = CalculateCentroids(test_poly3)
    print("Cx = " + str(test[0]))
    print("Cy = " + str(test[1]))
    print("Cz = " + str(test[2]))
