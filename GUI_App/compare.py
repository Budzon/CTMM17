# 7. Провести расчет для K равного 10 50 100 200 500 1000.
# 8. На одном графике построить графики времени расчета для различных
# методов в зависимости от числа точек. Подписи по оси X - число точек.
# Подписи по оси Y - время.
# На втором графике построить ускорение во времени работы программы по
# сравнению с самым медленным методом. Подписи по оси X - число точек.
# Подписи по оси Y - ускорение.

import time
import N_body as nb
import numpy as np
import random as random


def max_error(method, masses, init_pos, init_vel, dt, iterations):
    ideal, t = nb.solveNbodies(0, masses, init_pos, init_vel, dt, iterations)
    actual, t = nb.solveNbodies(method, masses, init_pos, init_vel, dt, iterations)

    return np.amax(np.abs(ideal - actual))


def data_for_sun_earth_moon():
    massSun = 1.98892e30  # kg
    massEarth = 5.972e24  # kg
    massMoon = 7.34767309e22  # kg
    orbitEarthSun = 1.496e11  # m
    orbitMoonEarth = 3.84467e8  # m
    velocityEarth = 2.9783e4  # m/s
    velocityMoon = 1022  # m/s

    return np.array([massSun, massEarth, massMoon]),\
           np.array([0, 0, orbitEarthSun, 0, orbitEarthSun + orbitMoonEarth, 0]),\
           np.array([0, 0, 0, velocityEarth, 0, velocityEarth + velocityMoon]),\
           60*60*24, 365*5


def solve_solar_system(method):
    data = data_for_sun_earth_moon()
    return nb.solveNbodies(method, data[0], data[1], data[2], data[3], data[4])


def timeit_solve(method, n_launches, masses, init_pos, init_vel, dt, iterations):
    t = 0
    for i in range(n_launches):
        t0 = time.time()
        nb.solveNbodies(method, masses, init_pos, init_vel, dt, iterations)
        t += time.time() - t0
    return t / n_launches


def generate_particles(n, dx=1e12):
    def gen_particle(cen):
        x = random.uniform(-2*dx/3, 2*dx/3)
        y = random.uniform(-2*dx/3, 2*dx/3)
        u = random.uniform(-dx**0.3, dx**0.3)
        v = random.uniform(-dx**0.3, dx**0.3)
        m = random.uniform(1, 1e5) * 1e22
        return [m, cen[0] + x, cen[1] + y, u, v]

    random.seed()
    cur_centre = np.zeros(2)
    particles = [gen_particle(cur_centre)]
    k = 1
    for i in range(n // 2):
        cur_centre[0] += k * dx * (-1)**k
        particles.append(gen_particle(cur_centre))
        cur_centre[1] += k * dx * (-1)**k
        particles.append(gen_particle(cur_centre))
        k += 1

    # Format for computation
    masses = []
    pos = []
    vel = []
    for p in particles:
        masses += p[:1]
        pos += p[1:3]
        vel += p[3:]

    return np.array(masses), np.array(pos), np.array(vel), 10 * abs((max(pos) - min(pos)) / max(vel))


m, p, v, t = generate_particles(100)
print(timeit_solve(7, 10, m, p, v, t / 500, 500))
