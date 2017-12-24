import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import threading
import multiprocessing as mp
import time
import cython_body
import pyopencl as cl


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
    
    # posvel = np.empty((times.size, half * 2))
    posvel = np.empty((2, half * 2))
    posvel[0] = np.concatenate((init_pos, init_vel))

    cur_accelerations = accelerations(masses, posvel[0, : half])
    for j in np.arange(iterations - 1) + 1:
        posvel[j % 2, :half] = posvel[(j-1) % 2, :half] + posvel[(j-1) % 2, half:] * dt + 0.5 * cur_accelerations * dt**2
        next_accelerations = accelerations(masses, posvel[j % 2, :half])
        posvel[j % 2, half:] = posvel[(j-1) % 2, half:] + 0.5 * (cur_accelerations + next_accelerations) * dt
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


def SolveNBodiesVerletOpenCL(masses, init_pos, init_vel, dt, iterations):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    masses_cl = np.array(masses, dtype=np.float)
    init_pos_cl = np.array(init_pos, dtype=np.float)
    init_vel_cl = np.array(init_vel, dtype=np.float)
    dt_cl = np.array(dt, dtype=np.float)
    iterations_cl = np.array(iterations, dtype=np.int)

    pos = np.empty((2, init_pos.size), dtype=np.float)
    vel = np.empty((2, init_vel.size), dtype=np.float)
    n_bodies_cl = np.array(masses.size, dtype=np.int)
    dimension_cl = np.array(init_pos.size // masses.size, dtype=np.int)
    accelerations = np.empty((2, init_pos.size), dtype=np.float)
    tmp_dimension_arr = np.empty(init_pos.size // masses.size, dtype=np.float)

    mf = cl.mem_flags
    masses_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=masses_cl)
    init_pos_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=init_pos_cl)
    init_vel_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=init_vel_cl)
    dt_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dt_cl)
    iterations_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=iterations_cl)
    pos_buf = cl.Buffer(ctx, mf.WRITE_ONLY, pos.nbytes)
    vel_buf = cl.Buffer(ctx, mf.WRITE_ONLY, vel.nbytes)
    n_bodies_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n_bodies_cl)
    dimension_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dimension_cl)
    accelerations_buf = cl.Buffer(ctx, mf.WRITE_ONLY, accelerations.nbytes)
    tmp_dimension_buf = cl.Buffer(ctx, mf.WRITE_ONLY, tmp_dimension_arr.nbytes)

    prg = cl.Program(ctx,
                     """
                     void acceleration(__global float *positions,
                                       __global float *masses_cl, 
                                       __global float *accelerations,
                                       __global float *tmp_dimension_arr, 
                                       int n_bodies, 
                                       int dimension,
                                       int second)
                     {
                         double G = 6.67e-11;
                         double norm, dx;
                         int shift = second * n_bodies * dimension;
                         for (int i = 0; i < n_bodies; ++i)
                         {
                             for (int d = 0; d < dimension; ++d)
                                 accelerations[shift + i*dimension + d] = 0;
                             for (int j = 0; j < n_bodies; ++j)
                                 if (i != j)
                                 {
                                    norm = 0;
                                    for (int d = 0; d < dimension; ++d)
                                    {
                                        dx = positions[shift + j*dimension + d] - positions[shift + i*dimension + d];
                                        norm += dx * dx;
                                        tmp_dimension_arr[d] = G * masses_cl[j] * dx;
                                    }
                                    norm = pow(norm, 1.5);
                                    for (int d = 0; d < dimension; ++d)
                                        accelerations[shift + i*dimension + d] += tmp_dimension_arr[d] / norm;
                                 }   
                         }                
                     }

                     __kernel void verlet_cl(__global float *masses_cl,
                                             __global float *init_pos_cl,
                                             __global float *init_vel_cl, 
                                             __global float *dt_cl, 
                                             __global int *iterations_cl,
                                             __global float *pos,
                                             __global float *vel,
                                             __global int *n_bodies_cl,
                                             __global int *dimension_cl,
                                             __global float *accelerations,
                                             __global float *tmp_dimension_arr)
                     {
                        int n_bodies = *n_bodies_cl, dimension = *dimension_cl, iterations = *iterations_cl;
                        float dt = *dt_cl;
                        int shift = n_bodies * dimension;

                        for (int coord = 0; coord < n_bodies * dimension; ++coord)
                        {
                            pos[coord] = init_pos_cl[coord];
                            vel[coord] = init_vel_cl[coord];
                        }

                        acceleration(pos, masses_cl, accelerations, tmp_dimension_arr, n_bodies, dimension, 0);
                        for (int t = 1; t < iterations; ++t)
                        {
                            for (int n = 0; n < n_bodies; ++n)
                                for (int d = 0; d < dimension; ++d)
                                    pos[(t%2)*shift + n*dimension + d] = pos[((t+1)%2)*shift + n*dimension + d] 
                                    + vel[((t+1)%2)*shift + n*dimension + d] * dt
                                    + 0.5 * accelerations[((t+1)%2)*shift + n*dimension + d] * dt * dt;

                            acceleration(pos, masses_cl, accelerations, tmp_dimension_arr, n_bodies, dimension, t % 2);

                            for (int n = 0; n < n_bodies; ++n)
                                for (int d = 0; d < dimension; ++d)
                                    vel[(t%2)*shift + n*dimension + d] = vel[((t+1)%2)*shift + n*dimension + d] 
                                    + 0.5 * (accelerations[n*dimension + d] + accelerations[shift + n*dimension + d]) * dt;  
                        }
                     }
                     """)
    try:
        prg.build()
    except:
        print("Error:")
        print(prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
        raise

    prg.verlet_cl(queue, (1,), None, masses_buf, init_pos_buf, init_vel_buf, dt_buf, iterations_buf, pos_buf, vel_buf,
                  n_bodies_buf, dimension_buf, accelerations_buf, tmp_dimension_buf)
    cl.enqueue_read_buffer(queue, pos_buf, pos).wait()
    cl.enqueue_read_buffer(queue, vel_buf, vel).wait()
    return np.concatenate((pos, vel), 1), np.arange(iterations) * dt


def solveNbodies(method, masses, init_pos, init_vel, dt, iterations):
    methods = {0: solveNbodiesOdeint,
               1: solveNbodiesVerlet,
               2: solveNbodiesVerletThreading,
               3: solveNbodiesVerletMultiprocessing,
               4: cython_body.SolveNbodiesVerletCython_notm_nomp,
               5: cython_body.SolveNbodiesVerletCython_notm_mp,
               6: cython_body.SolveNbodiesVerletCython_tm_nomp,
               7: cython_body.SolveNbodiesVerletCython_tm_mp,
               8: SolveNBodiesVerletOpenCL}
    return methods[method](masses, init_pos, init_vel, dt, iterations)


def sunEarthMoon(method):
    masses = np.array([massSun, massEarth, massMoon])
    init_pos = np.array([0, 0, orbitEarthSun, 0, orbitEarthSun + orbitMoonEarth, 0])
    init_vel = np.array([0, 0, 0, velocityEarth, 0, velocityEarth + velocityMoon])

    return solveNbodies(method, masses, init_pos, init_vel, 60*60*24, 365*5)


# t = time.time()
# posvel, times = sunEarthMoon(4)
# print(time.time() - t)
# plt.plot(posvel[:, 0], posvel[:, 1], 'orange')
# plt.plot(posvel[:, 2], posvel[:, 3], 'cyan')
# plt.plot(posvel[:, 4], posvel[:, 5], 'red', linewidth=0.5)
# plt.show()
