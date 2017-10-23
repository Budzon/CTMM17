import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sympy import Symbol, solve, lambdify, Matrix

tol = 1e-12

k1 = Symbol('k1')
k1m = Symbol('k1m')
k3 = Symbol('k3')
k3m = Symbol('k3m')
k2 = Symbol('k2')
x = Symbol('x')  
y = Symbol('y')

k1val = 0.12
k1mval = 0.01
k3val = 0.0032
k3mval = 0.02
k2val = 0.95

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

X = np.arange(0, 1, 1e-5)
 
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
    plt.plot(k1_at_bifurcation_np, x_at_bifurcation_np, linestyle='', marker ='^')
    plt.plot(k1_at_bifurcation_np, y_of_x(x_at_bifurcation_np, k3val, k3mval), linestyle='', marker ='o')
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

	plt.plot(k1Trace_of_x(X, k3val, k3mval, k1mval), k2Trace_of_x(X, k3val, k3mval, k1mval), label = 'neutral')
	plt.plot(k1Det_of_x(X, k3val, k3mval, k1mval), k2Det_of_x(X, k3val, k3mval, k1mval), linestyle = '--', linewidth = 0.5, label = 'multiplicity')
	plt.xlim([-0.1, 0.6])
	plt.ylim([-0.1, 4])
	plt.legend(loc = 1)
	return

k1m_range = [1e-3, 5e-3, 1e-2, 1.5e-2, 2e-2]
k3m_range = [5e-4, 1e-3, 2e-3, 3e-3, 4e-3]

twoParameterAnalysisK1K2(k1m_range[1], 0.0032, k3m_range[0])
plt.grid()
plt.show()