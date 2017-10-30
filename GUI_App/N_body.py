import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import figure
import threading


massSun = 1.98892e30 #kg
massEarth = 5.972e24 #kg
massMoon = 7.34767309e22 #kg
orbitEarthSun = 1.496e11 #m
orbitMoonEarth = 3.84467e8 #m
velocityEarth = 2.9783e4 # m/s
velocityMoon = 1022 # m/s

G = 6.67408e-11

def acceleration(body, masses, positions):
    global G
    dimension = positions.size // masses.size
    
    res = np.zeros(dimension)
    for j in range(masses.size):
        if (body != j):
            displacement = positions[j * dimension : (j+1) * dimension] - positions[body * dimension : (body+1) * dimension]
            res += G*masses[j] * displacement / np.linalg.norm(displacement, 2)**3
    return res

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

def solveNbodiesVerlet(masses, initPos, initVel, dt, iterations):
    times = np.arange(iterations) * dt
    half = initPos.size
    
    posvel = np.empty((times.size, half * 2))
    posvel[0] = np.concatenate((initPos, initVel))

    curAccelerations = accelerations(masses, posvel[0, : half])
    for j in np.arange(iterations - 1) + 1:
        posvel[j, : half] = posvel[j-1, : half] + posvel[j-1, half :] * dt + 0.5 * curAccelerations * dt**2
        nextAccelerations = accelerations(masses, posvel[j, : half])
        posvel[j, half :] = posvel[j-1, half :] + 0.5 * (curAccelerations + nextAccelerations) * dt
        curAccelerations = nextAccelerations

    return posvel, times

def solveNbodiesVerletThreading(masses, initPos, initVel, dt, iterations):
    times = np.arange(iterations) * dt
    half = initPos.size
    dimension = len(initPos) // len(masses)

    posvel = np.empty((times.size, half * 2))
    posvel[0] = np.concatenate((initPos, initVel))

    def solveForOneBody(body, event_wait, event_set):
        curAcceleration = acceleration(body, masses, posvel[0, : half])
        for j in np.arange(iterations - 1) + 1:
            event_wait.wait()
            event_wait.clear()
            posvel[j, body*dimension : (body+1)*dimension] = (posvel[j-1, body*dimension : (body+1)*dimension] 
                                                              + posvel[j-1, half + body*dimension : half + (body+1)*dimension] * dt 
                                                              + 0.5 * curAcceleration * dt**2)
            nextAcceleration = acceleration(body, masses, posvel[j, : half])
            posvel[j, half + body*dimension : half + (body+1)*dimension :] = (posvel[j-1, half + body*dimension : half + (body+1)*dimension] 
                                                                              + 0.5 * (curAcceleration + nextAcceleration) * dt)
            curAcceleration = nextAcceleration
            event_set.set()

    events = []
    for body in masses:
        events.append(threading.Event())
        events[-1].clear()
    events[0].set()

    threads = []
    for body in range(masses.size):
        threads.append(threading.Thread(target = solveForOneBody, args=(body, events[body-1], events[body])))
        threads[-1].start()

    for thread in threads:
        thread.join()

    return posvel, times

def solveNbodiesOdeint(masses, initPos, initVel, dt, iterations):
    def rhs(xv, t):
        res = np.zeros(xv.shape)
        res[xv.size // 2 :] = accelerations(masses, xv[: xv.size // 2])
        res[: xv.size // 2] = xv[xv.size // 2 :]
        return res

    times = np.arange(iterations) * dt
    res = odeint(rhs, np.concatenate((initPos, initVel)), times)
    return res, times

def sunEarthMoon(method):
    masses = np.array([massSun, massEarth, massMoon])
    initPos = np.array([0, 0, orbitEarthSun, 0, orbitEarthSun + orbitMoonEarth, 0])
    initVel = np.array([0, 0, 0, velocityEarth, 0, velocityEarth + velocityMoon])

    methods = {0: solveNbodiesOdeint, 1: solveNbodiesVerlet, 2: solveNbodiesVerletThreading, 3: solveNbodiesOdeint, 4: solveNbodiesOdeint}
    print(methods[method].__name__)
    return methods[method](masses, initPos, initVel, 60*60*24, 365) #120 : 240000
