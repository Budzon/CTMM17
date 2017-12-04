import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import threading
import multiprocessing as mp
import time
import cython_body


massSun = 1.98892e30  # kg
massEarth = 5.972e24  # kg
massMoon = 7.34767309e22  # kg
orbitEarthSun = 1.496e11  # m
orbitMoonEarth = 3.84467e8  # m
velocityEarth = 2.9783e4  # m/s
velocityMoon = 1022  # m/s

G = 6.67408e-11


def acceleration(body, masses, positions):
    # global G
    dimension = positions.size // masses.size

    res = np.zeros(dimension)
    for j in range(masses.size):
        if body != j:
            displacement = positions[j*dimension:(j+1) * dimension] - positions[body*dimension:(body+1)*dimension]
            res += G*masses[j] * displacement / np.linalg.norm(displacement, 2)**3
    return res


def acceleration_no_np(body, masses, positions):
    dimension = len(positions) // len(masses)

    res = np.zeros(dimension)
    displacement = np.empty(dimension)
    for j in range(masses.size):
        if body != j:
            for k in range(dimension):
                displacement[k] = positions[j * dimension + k] - positions[body * dimension + k]
            res += G * masses[j] * displacement / np.linalg.norm(displacement, 2)**3
    return res


def accelerations(masses, positions):
    # global G
    dimension = positions.size // masses.size
    res = np.zeros(positions.size)
    for i in range(masses.size):
        for j in range(masses.size):
            if i != j:
                displacement = positions[j*dimension:(j+1)*dimension] - positions[i*dimension:(i+1)*dimension]
                res[i*dimension:(i+1)*dimension] += G*masses[j] * displacement / np.linalg.norm(displacement, 2)**3
    return res


def solveNbodiesVerlet(masses, init_pos, init_vel, dt, iterations):
    times = np.arange(iterations) * dt
    half = init_pos.size
    
    posvel = np.empty((times.size, half * 2))
    posvel[0] = np.concatenate((init_pos, init_vel))

    cur_accelerations = accelerations(masses, posvel[0, : half])
    for j in np.arange(iterations - 1) + 1:
        posvel[j, :half] = posvel[j-1, :half] + posvel[j-1, half:] * dt + 0.5 * cur_accelerations * dt**2
        next_accelerations = accelerations(masses, posvel[j, :half])
        posvel[j, half:] = posvel[j-1, half:] + 0.5 * (cur_accelerations + next_accelerations) * dt
        cur_accelerations = next_accelerations

    return posvel, times


def solveNbodiesVerletThreading(masses, init_pos, init_vel, dt, iterations):
    times = np.arange(iterations) * dt
    half = init_pos.size
    dimension = len(init_pos) // len(masses)

    posvel = np.empty((times.size, half * 2))
    posvel[0] = np.concatenate((init_pos, init_vel))

    def solveForOneBody(body, event_wait, event_set):
        cur_acceleration = acceleration(body, masses, posvel[0, :half])
        for j in np.arange(iterations - 1) + 1:
            event_wait.wait()
            event_wait.clear()
            posvel[j, body*dimension:(body+1)*dimension] = \
                (posvel[j-1, body*dimension:(body+1)*dimension]
                 + posvel[j-1, half+body*dimension:half+(body+1)*dimension] * dt
                 + 0.5 * cur_acceleration * dt**2)
            next_acceleration = acceleration(body, masses, posvel[j, :half])
            posvel[j, half + body*dimension:half + (body+1)*dimension] = \
                (posvel[j-1, half + body*dimension:half + (body+1)*dimension]
                 + 0.5 * (cur_acceleration + next_acceleration) * dt)
            cur_acceleration = next_acceleration
            event_set.set()

    events = []
    for body in masses:
        events.append(threading.Event())
        events[-1].clear()
    events[0].set()

    threads = []
    for body in range(masses.size):
        threads.append(threading.Thread(target=solveForOneBody, args=(body, events[body-1], events[body])))
        threads[-1].start()

    for thread in threads:
        thread.join()

    return posvel, times


def solveNbodiesVerletMultiprocessing(masses, init_pos, init_vel, dt, iterations):
    def solveForOneBody(q, q_out, init_vel, shared_pos, body, events1, events2):
        dimension = len(shared_pos) // len(masses)
        my_cur_pos = np.array(shared_pos[body*dimension:(body+1)*dimension])
        cur_acceleration = acceleration_no_np(body, masses, shared_pos)

        result = np.empty((iterations, dimension * 2))
        result[0, :] = np.concatenate((my_cur_pos, init_vel[body*dimension:(body+1)*dimension]))

        for j in np.arange(iterations - 1) + 1:
            result[j, :dimension] = \
                (my_cur_pos
                 + result[j-1, dimension:] * dt
                 + 0.5 * cur_acceleration * dt**2)

            q.put([body, result[j, :dimension]])
            events1[body].set()

            if body == 0:
                for i in range(len(masses)):
                    events1[i].wait()
                    events1[i].clear()
                for i in range(len(masses)):
                    tmp = q.get()
                    shared_pos[tmp[0]*dimension:(tmp[0]+1)*dimension] = tmp[1]
                for i in range(len(masses)):
                    events2[i].set()
            else:
                events2[body].wait()
                events2[body].clear()

            my_cur_pos = np.array(shared_pos[body * dimension:(body + 1) * dimension])
            next_acceleration = acceleration_no_np(body, masses, shared_pos)

            result[j, dimension:] = \
                (result[j-1, dimension:]
                 + 0.5 * (cur_acceleration + next_acceleration) * dt)
            cur_acceleration = next_acceleration
        q_out.put([body, result])

    if __name__ == '__main__':
        times = np.arange(iterations) * dt

        posvel = np.empty((times.size, init_pos.size * 2))
        shared_pos = mp.Array('d', init_pos)

        events1 = []
        events2 = []
        for body in masses:
            events1.append(mp.Event())
            events2.append(mp.Event())
            events1[-1].clear()
            events2[-1].clear()

        q = mp.Queue()
        q_out = mp.Queue()
        processes = []
        for body in range(masses.size):
            processes.append(mp.Process(target=solveForOneBody, args=(q, q_out, init_vel, shared_pos, body, events1, events2)))
            processes[-1].start()

        dim = init_pos.size // len(masses)
        for i in range(len(masses)):
            tmp = q_out.get()
            posvel[:, tmp[0]*dim:(tmp[0]+1)*dim] = tmp[1][:, :dim]
            posvel[:, init_pos.size + tmp[0]*dim: init_pos.size + (tmp[0]+1)*dim] = tmp[1][:, dim:]

        for process in processes:
            process.join()

        return posvel, times


def solveNbodiesOdeint(masses, init_pos, init_vel, dt, iterations):
    def rhs(xv, t):
        res = np.zeros(xv.shape)
        res[xv.size//2:] = accelerations(masses, xv[:xv.size//2])
        res[:xv.size//2] = xv[xv.size//2:]
        return res

    times = np.arange(iterations) * dt
    res = odeint(rhs, np.concatenate((init_pos, init_vel)), times)
    return res, times


def solveNbodies(method, masses, init_pos, init_vel, dt, iterations):
    methods = {0: solveNbodiesOdeint,
               1: solveNbodiesVerlet,
               2: solveNbodiesVerletThreading,
               3: solveNbodiesVerletMultiprocessing,
               4: cython_body.SolveNbodiesVerletCython_notm_nomp,
               5: cython_body.SolveNbodiesVerletCython_notm_mp,
               6: cython_body.SolveNbodiesVerletCython_tm_nomp,
               7: cython_body.SolveNbodiesVerletCython_tm_mp}
    return methods[method](masses, init_pos, init_vel, dt, iterations)


def sunEarthMoon(method):
    masses = np.array([massSun, massEarth, massMoon])
    init_pos = np.array([0, 0, orbitEarthSun, 0, orbitEarthSun + orbitMoonEarth, 0])
    init_vel = np.array([0, 0, 0, velocityEarth, 0, velocityEarth + velocityMoon])

    return solveNbodies(method, masses, init_pos, init_vel, 60*60*24, 365*5)


t = time.time()
posvel, times = sunEarthMoon(4)
print(time.time() - t)
# plt.plot(posvel[:, 0], posvel[:, 1], 'orange')
# plt.plot(posvel[:, 2], posvel[:, 3], 'cyan')
# plt.plot(posvel[:, 4], posvel[:, 5], 'red', linewidth=0.5)
# plt.show()
