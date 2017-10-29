import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sympy import Symbol, solve, lambdify, Matrix

tol = 1e-12

k1 = Symbol('k1', Positive=True)
k1m = Symbol('k1m', Positive=True)
k3 = Symbol('k3', Positive=True)
k3m = Symbol('k3m', Positive=True)
k2 = Symbol('k2', Positive=True)
x = Symbol('x', Positive=True)  
y = Symbol('y', Positive=True)

k1val = 0.09# 0.12
k1mval = 0.01
k3val = 0.0032
k3mval = 0.002
k2val = 1.#0.95

eq1 = k1*(1 - x - 2*y) - k1m*x - k3*x*(1 - x - 2*y) + k3m*y - k2*x*(1 - x - 2*y)**2
eq2 = k3*x*(1 - x - 2*y) - k3m*y

solution = solve([eq1, eq2], y, k1)
ySolution = solution[0][0]
k1Solution = solution[0][1]

A = Matrix([eq1, eq2])
var_vector = Matrix([x, y])
jacA = A.jacobian(var_vector)
detA = jacA.det()
traceA = jacA.trace()
discriminantA = traceA**2-4*detA

X = np.arange(0, 1, 1e-3)
 
def oneParemeterAnalysisK1(k1mval, k3mval):
    k1_of_x = lambdify((x, k1m, k3, k3m, k2), k1Solution)
    y_of_x = lambdify((x, k3, k3m), ySolution)
    detA_of_x = lambdify((x, k1m, k3, k3m, k2), detA.subs(y, ySolution).subs(k1, k1Solution))

    detA_of_X = detA_of_x(X, k1mval, k3val, k3mval, k2val)
    x_at_bifurcation = []
    for i in range(len(X) - 1):
        if (detA_of_X[i] * detA_of_X[i + 1] < tol):
        	x_at_bifurcation.append(X[i] - detA_of_X[i] * (X[i + 1] - X[i]) / (detA_of_X[i + 1] - detA_of_X[i]))

    x_at_bifurcation_np = np.array(x_at_bifurcation)
    k1_at_bifurcation_np = np.array(k1_of_x(x_at_bifurcation_np, k1mval, k3val, k3mval, k2val))

    plt.plot(k1_of_x(X, k1mval, k3val, k3mval, k2val), X, label='$x(k_1)$')
    plt.plot(k1_of_x(X, k1mval, k3val, k3mval, k2val), y_of_x(X, k3val, k3mval), label='$y(k_1)$') 
    plt.plot(k1_at_bifurcation_np, x_at_bifurcation_np, linestyle='', marker ='x', color='k')
    plt.plot(k1_at_bifurcation_np, y_of_x(x_at_bifurcation_np, k3val, k3mval), linestyle='', marker ='o', color='k')
    plt.xlim([0.0, 0.7])
    plt.ylim([-0.05, 1.1])
    plt.legend(loc=2)
    return k1_at_bifurcation_np

def twoParameterAnalysisK1K2(k1mval, k3val, k3mval):
	# Neutral lines
	k1TraceSolution = solve(traceA.subs(y, ySolution), k1)[0]
	k2JointTraceSolution = solve(k1TraceSolution - k1Solution, k2)[0]
	k1JointTraceSolution = k1Solution.subs(k2, k2JointTraceSolution)

	k1Trace_of_x = lambdify((x, k3, k3m, k1m), k1JointTraceSolution)
	k2Trace_of_x = lambdify((x, k3, k3m, k1m), k2JointTraceSolution)

	# Multiplicity lines
	k1DetSolution = solve(detA.subs(y, ySolution), k1)[0]
	k2JointDetSolution = solve(k1DetSolution - k1Solution, k2)[0]
	k1JointDetSolution = k1Solution.subs(k2, k2JointDetSolution)

	k1Det_of_x = lambdify((x, k3, k3m, k1m), k1JointDetSolution)
	k2Det_of_x = lambdify((x, k3, k3m, k1m), k2JointDetSolution)

	XX = X[k1Det_of_x(X, k3val, k3mval, k1mval) > 0]

	plt.plot(k1Trace_of_x(XX, k3val, k3mval, k1mval), k2Trace_of_x(XX, k3val, k3mval, k1mval), linestyle = '--',  linewidth = 1.5, label = 'neutral')
	plt.plot(k1Det_of_x(XX, k3val, k3mval, k1mval), k2Det_of_x(XX, k3val, k3mval, k1mval), linewidth = .8, label = 'multiplicity')
	plt.xlim([-0.0, 0.6])
	plt.ylim([-0.0, 8.5])
	plt.xlabel(r'$k_1$')
	plt.ylabel(r'$k_2$')
	plt.legend(loc = 0)
	return

def solveSystem(init, k1val, k1mval, k3val, k3mval, k2val, dt, iterations):
	f1 = lambdify((x, y, k1, k1m, k3, k3m, k2), eq1)
	f2 = lambdify((x, y, k3, k3m), eq2)

	def rhs(xy, t):
		return [f1(xy[0], xy[1], k1val, k1mval, k3val, k3mval, k2val), f2(xy[0], xy[1], k3val, k3mval)]
	times = np.arange(iterations) * dt
	return odeint(rhs, init, times), times

def streamplot(k1val, k1mval, k3val, k3mval, k2val):
	f1 = lambdify((x, y, k1, k1m, k3, k3m, k2), eq1)
	f2 = lambdify((x, y, k3, k3m), eq2)

	Y, X = np.mgrid[0:.5:1000j, 0:1:2000j]
	U = f1(X, Y, k1val, k1mval, k3val, k3mval, k2val)
	V = f2(X, Y, k3val, k3mval)
	velocity = np.sqrt(U*U + V*V)
	plt.streamplot(X, Y, U, V, density = [2.5, 0.8], color=velocity)
	plt.xlabel('x')
	plt.ylabel('y')

k1m_range = [1e-3, 5e-3, 1e-2, 1.5e-2, 2e-2]
k3m_range = [5e-4, 1e-3, 2e-3, 3e-3, 4e-3]

# twoParameterAnalysisK1K2(k1m_range[2], k3val, k3m_range[1])
res, times = solveSystem([0.38, 0.22], k1val, k1m_range[2], k3val, k3m_range[1], k2val, 1e-2, 1e6)
ax = plt.subplot(211)
plt.plot(times, res[:, 0])
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('x')
plt.grid()
ax1 = plt.subplot(212, sharex = ax)
plt.plot(times, res[:, 1], color='orange')
plt.xlabel('t')
plt.ylabel('y')
plt.grid()
plt.show()