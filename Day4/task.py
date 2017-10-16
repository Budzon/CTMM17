import numpy as np
import scipy as sp
from scipy.integrate import odeint

def accelerations(masses, positions):
    G = 6.67408e-11
    res = np.zeros(masses.shape)
    with np.errstate(divide='ignore'):
        for i in np.arange(res.size):
            distances = positions - positions[i]
            res[i] = np.sum(distances / np.abs(distances)**3 * (G*masses))
    return res

def solveNbodiesVerle(masses, positions, velocities, endTime):
    times, dt = np.linspace(0, endTime, 1e-2)

    curAccelerations = accelerations(masses, positions)
    for t in times:
        positions += velocities * dt  + 0.5 * curAccelerations * dt**2
        nextAccelerations = accelerations(masses, positions)
        velocities += 0.5 * (curAccelerations + nextAccelerations) * dt
        curAccelerations = nextAccelerations

    return positions, velocities

def solveNbodiesOdeint(masses, positions, velocities, endTime):
    f = lambda xv: np.concatenate(xv[xv.size / 2 : ], accelerations(masses, xv[ : xv.size / 2]))
    times, dt = np.linspace(0, endTime, 1e-2)
    res = odeint(f, np.concatenate(positions, velocities), times)
    return res

print(accelerations(np.array([1e6,1e6,1e6]), np.array([[-1, 0, 0], [0,1,0], [1,0,0]])))
