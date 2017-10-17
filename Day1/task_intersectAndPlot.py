EPS = 1e-6


def isPositive(a, tol=EPS):
    return a > tol


def isNegative(a, tol=EPS):
    return a < -tol


def isZero(a, tol=EPS):
    return (not isPositive(a, EPS)) and (not isNegative(a, EPS))


def vecIsZero(v, tol=EPS):
    return all(isZero(a, EPS) for a in v)


def dotProduct(u, v):
    return sum(w[0] * w[1] for w in zip(u, v))


def crossProduct(u, v):
    return [u[1] * v[2] - u[2] * v[1], -u[0] * v[2] + u[2] * v[0], u[0] * v[1] - u[1] * v[0]]


def difference(u, v):
    return [w[0] - w[1] for w in zip(u, v)]


def byScalar(u, a):
    return [a * x for x in u]


def areParallel(normal1, normal2):
    return vecIsZero(crossProduct(normal1, normal2))


def areCoplanar(point1, point2, normal):
    return isZero(dotProduct(point1, normal) - dotProduct(point2, normal))


def pointsInNegativeHalfspace(normal, refPoint, points):
    return all(isNegative(dotProduct(difference(point, refPoint), normal)) for point in points)


def pointInPositiveHalfspaces(normals, refPoints, point):
    return all(not isNegative(dotProduct(difference(point, refPoint), normal)) for (normal, refPoint) in
               zip(normals, refPoints))


def intersectEdgeAndPlane(normal, pointInPlane, point, direction):
    denom = dotProduct(direction, normal)
    if (isZero(denom)):
        return False, [0, 0, 0]
    t = dotProduct(difference(pointInPlane, point), normal) / denom
    if (isNegative(t) or isPositive(t - 1)):
        return False, [0, 0, 0]
    return True, difference(point, byScalar(direction, -t))


def testPoints(normal, pointInPlane, triag, edges):
    temp = [intersectEdgeAndPlane(normal, pointInPlane, point, direction) for (point, direction) in zip(triag, edges)]
    return [intersection[1] for intersection in temp if intersection[0]]


def doTrianglesIntersect3D(triag1, triag2):
    edges1 = [difference(triag1[1], triag1[0]), difference(triag1[2], triag1[1]), difference(triag1[0], triag1[2])]
    edges2 = [difference(triag2[1], triag2[0]), difference(triag2[2], triag2[1]), difference(triag2[0], triag2[2])]
    normalPlane1 = crossProduct(edges1[0], edges1[1])
    normalPlane2 = crossProduct(edges2[0], edges2[1])
    positiveNormals1 = [crossProduct(normalPlane1, edge) for edge in edges1]
    positiveNormals2 = [crossProduct(normalPlane2, edge) for edge in edges2]
    planes1 = [(lambda p: dotProduct(p, normal) - dotProduct(refPoint, normal)) for (refPoint, normal) in
               zip(triag1, positiveNormals1)]
    planes2 = [(lambda p: dotProduct(p, normal) - dotProduct(refPoint, normal)) for (refPoint, normal) in
               zip(triag2, positiveNormals2)]

    if (areParallel(normalPlane1, normalPlane2)):
        if (areCoplanar(triag1[0], triag2[0], normalPlane1)):
            return not (any(pointsInNegativeHalfspace(normal, refPoint, triag1) for (normal, refPoint) in
                            zip(positiveNormals2, triag2)) or any(
                pointsInNegativeHalfspace(normal, refPoint, triag2) for (normal, refPoint) in
                zip(positiveNormals1, triag1)))
        else:
            return False
    else:
        return (any(pointInPositiveHalfspaces(positiveNormals1, triag1, point) for point in
                    testPoints(normalPlane1, triag1[0], triag2, edges2))) or (any(
            pointInPositiveHalfspaces(positiveNormals2, triag2, point) for point in
            testPoints(normalPlane2, triag2[0], triag1, edges1)))


import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as plt
import scipy as sp


def plotTriangle(ax, tr):
    tri = a3.art3d.Poly3DCollection([tr])
    tri.set_color(colors.rgb2hex(sp.rand(3)))
    tri.set_edgecolor('k')
    ax.add_collection3d(tri)


trianglePairs = []

# 1
p1 = [0, 0, 0]
p2 = [0, 5, 0]
p3 = [6, 5, 0]
tr1 = [p1, p2, p3]
q1 = [1, 4, 0]
q2 = [2, 4, 0]
q3 = [2, 3, 0]
tr2 = [q1, q2, q3]
trianglePairs.append((tr1, tr2))

# 2
p1 = [-1, 0, 0]
p2 = [0, -1, 0]
p3 = [0, 0, 0]
tr1 = [p1, p2, p3]
q1 = [0, 0, 0]
q2 = [0, 3, 0]
q3 = [5, 0, 0]
tr2 = [q1, q2, q3]
trianglePairs.append((tr1, tr2))

# 3
p1 = [-1, 0, 0]
p2 = [0, 2, 0]
p3 = [0, 0, 0]
tr1 = [p1, p2, p3]
q1 = [0, 0, 0]
q2 = [5, 0, 0]
q3 = [0, 4, 0]
tr2 = [q1, q2, q3]
trianglePairs.append((tr1, tr2))

# 4
p1 = [0, 0, 0]
p2 = [0, 2, 0]
p3 = [1, 0, 0]
tr1 = [p1, p2, p3]
q1 = [0, -1, 0]
q2 = [0, 3, 0]
q3 = [7, -1, 0]
tr2 = [q1, q2, q3]
trianglePairs.append((tr1, tr2))

# 5
p1 = [0, 0, 0]
p2 = [0, 4, 0]
p3 = [4, 0, 0]
tr1 = [p1, p2, p3]
q1 = [1, 2, 0]
q2 = [1, 1, -3]
q3 = [0.5, 2, -2]
tr2 = [q1, q2, q3]
trianglePairs.append((tr1, tr2))

# 6
p1 = [0, 0, 0]
p2 = [0, 4, 0]
p3 = [4, 0, 0]
tr1 = [p1, p2, p3]
q1 = [4, 0, 0]
q2 = [1, 1, -3]
q3 = [0.5, 2, -2]
tr2 = [q1, q2, q3]
trianglePairs.append((tr1, tr2))

# 7
p1 = [0, 0, 0]
p2 = [0, 4, 0]
p3 = [4, 0, 0]
tr1 = [p1, p2, p3]
q1 = [1, 2, 0]
q2 = [2, 1, 0]
q3 = [0.5, 2, -2]
tr2 = [q1, q2, q3]
trianglePairs.append((tr1, tr2))

# 8
p1 = [0, 0, 0]
p2 = [0, 4, 0]
p3 = [4, 0, 0]
tr1 = [p1, p2, p3]
q1 = [-1, 2, 2]
q2 = [0, 2, 2]
q3 = [0, 0, -2]
tr2 = [q1, q2, q3]
trianglePairs.append((tr1, tr2))

# 9
p1 = [0, 0, 0]
p2 = [0, 4, 0]
p3 = [5, 0, 0]
tr1 = [p1, p2, p3]
q1 = [1, 1, 2]
q2 = [5, 6, -2]
q3 = [3, -4, -1]
tr2 = [q1, q2, q3]
trianglePairs.append((tr1, tr2))

for pair in trianglePairs:
    print(doTrianglesIntersect3D(pair[0], pair[1]))
    ax = a3.Axes3D(plt.figure())
    plotTriangle(ax, pair[0])
    plotTriangle(ax, pair[1])
    plt.show()





