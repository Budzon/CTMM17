from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri
from math import pi,sinh,cosh
from mshr import *


tol = 1e-14
box = Rectangle(Point(0, 0), Point(pi,pi))
mesh = generate_mesh(box, 256)
V = FunctionSpace(mesh, 'P', 1)

u_left = Constant(0.0)
u_right = Expression('cos(2*x[1])', degree=1)

g_left = Expression('sin(2*x[0])', degree=1)
g_right = Expression('sin(3*x[0])', degree=1)


def boundary_left(x, on_boundary):
    return on_boundary and near(x[0], 0, tol)


def boundary_right(x, on_boundary):
    return on_boundary and near(x[0], pi, tol)


bc_left = DirichletBC(V, u_left, boundary_left)
bc_right = DirichletBC(V, u_right, boundary_right)
bcs = [bc_left, bc_right]


class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0, tol)


class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], pi, tol)


top = Top()
bottom = Bottom()
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
top.mark(boundaries, 1)
bottom.mark(boundaries, 2)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v)) * dx
L = g_right * v * ds(1) - g_left * v * ds(2)

fem_solution = Function(V)
solve(a == L, fem_solution, bcs)

paper_solution = Expression('sinh(2*x[0])*cos(2*x[1])/sinh(2*pi) + cosh(3*x[1])*sin(3*x[0])/3/sinh(3*pi) - cosh(2*(x[1] - pi))*sin(2*x[0])/2/sinh(2*pi)', degree=2)
error_L2 = errornorm(paper_solution, fem_solution, 'L2')
vertex_values_fem = fem_solution.compute_vertex_values(mesh)
vertex_values_paper = paper_solution.compute_vertex_values(mesh)
error_max = np.max(np.abs(vertex_values_fem - vertex_values_paper))
print('error_L2 =', error_L2)
print('error_max =', error_max)

n = mesh.num_vertices()
d = mesh.geometry().dim()
mesh_coordinates = mesh.coordinates().reshape((n, d))
triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)

plt.figure(1)
plt.title("Analytical")
zfaces = np.asarray([paper_solution(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.colorbar()
plt.savefig('fem_analytical.pdf')

plt.figure(2)
plt.title("Numerical")
zfaces = np.asarray([fem_solution(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.colorbar()
plt.savefig('fem_numerical.pdf')


