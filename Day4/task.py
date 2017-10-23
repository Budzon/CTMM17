import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

massSun = 1.98892e30 #kg
massEarth = 5.9742e24 #kg
massMoon = 7.36e22 #kg
orbitEarthSun = 1.5e11 #m
orbitMoonEarth = 3.85e8 #m

G = 6.67408e-11# * massEarth / orbitMoonEarth**3

def accelerations(masses, positions):
    global G
    dimension = positions.size // masses.size
    res = np.zeros(positions.size)
    for i in range(masses.size):
        for j in range(masses.size):
            if (i != j):
                displacement = positions[j * dimension : (j+1) * dimension] - positions[i * dimension : (i+1) * dimension]
                res[i * dimension : (i+1) * dimension] += G*masses[j] * displacement / np.linalg.norm(displacement, 2)**3
    return res
    # with np.errstate(divide='ignore'):
    #     for i in range(masses.size):
    #         displacements = np.delete(positions, i, axis = 0) - positions[i]
    #         distances = np.linalg.norm(displacements, 2, axis = 1)
    #         res[i] = np.sum(displacements / distances**3 * (G*np.delete(masses,i)), axis = 0)
    # return res

def solveNbodiesVerle(masses, initPosVel, endTime, dt):
    times = np.arange(0, endTime, dt)
    print(times.size)
    dimension = initPosVel.shape[1] // 2
    
    posvel = np.empty((times.size, masses.size, dimension * 2))
    posvel[0] = initPosVel

    curAccelerations = accelerations(masses, posvel[0, :, : dimension])
    for j in np.arange(times.size - 1) + 1:
        print(j)
        posvel[j, :, : dimension] = posvel[j-1, :, : dimension] + posvel[j-1, :, dimension :] * dt + 0.5 * curAccelerations * dt**2
        # positions += velocities * dt  + 0.5 * curAccelerations * dt**2
        nextAccelerations = accelerations(masses, posvel[j, :, : dimension])
        posvel[j, :, dimension :] = posvel[j-1, :, dimension :] + 0.5 * (curAccelerations + nextAccelerations) * dt
        # velocities += 0.5 * (curAccelerations + nextAccelerations) * dt
        curAccelerations = nextAccelerations

    return posvel, times

def solveNbodiesOdeint(masses, initPos, initVel, endTime, dt):
    def rhs(xv, t):
        res = np.zeros(xv.shape)
        res[xv.size // 2 :] = accelerations(masses, xv[: xv.size // 2])
        res[: xv.size // 2] = xv[xv.size // 2 :]
        # print(res)
        return res
    def f(xv,t):
        res = np.zeros(xv.shape)
        res[0] = xv[1]
        res[1] =-xv[0]
        return res

    times = np.arange(0, endTime, dt)
    res = odeint(rhs, np.concatenate((initPos, initVel)), times)
    return res, times #.reshape(times.size, masses.size, 2 * dimension), times

masses = np.array([1e8, 1])
initPos = [0, 0, 50, 0]
initVel = [0, 0, 1.2, 0]

earthMoon, times = solveNbodiesOdeint(masses, initPos, initVel, 10000, 5)
plt.plot(earthMoon[:, 2], earthMoon[:, 3])
plt.show()
