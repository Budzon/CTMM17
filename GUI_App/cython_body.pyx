cimport cython
cimport numpy as np
import numpy as np
from cython.parallel import prange, parallel

from libc.math cimport sqrt, pow

cdef double G = 6.67408e-11


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] accelerations_notm_nomp(np.ndarray[double, ndim=1] masses,
                                                        np.ndarray[double, ndim=1] positions):
    cdef:
        int n_bodies = masses.size
        int dimension = <int>positions.size / <int>masses.size
        double norm
        np.ndarray res = np.zeros(positions.size)
        np.ndarray dx = np.zeros(dimension)
        int i, j

    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                dx = positions[j*dimension:(j+1)*dimension] - positions[i*dimension:(i+1)*dimension]
                norm = pow(np.linalg.norm(dx), 3)
                res[i*dimension:(i+1)*dimension] += G*masses[j] * dx / norm
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def SolveNbodiesVerletCython_notm_nomp(np.ndarray[double, ndim=1] masses,
                                       np.ndarray[double, ndim=1] init_pos,
                                       np.ndarray[double, ndim=1] init_vel,
                                       double dt,
                                       int iterations):
    cdef:
        times = np.arange(iterations) * dt
        int half = init_pos.size
        np.ndarray[double, ndim=2] pos = np.empty((times.size, half))
        np.ndarray[double, ndim=2] vel = np.empty((times.size, half))
        int j

    pos[0] = init_pos
    vel[0] = init_vel

    cdef np.ndarray[double, ndim=1] cur_accelerations = accelerations_notm_nomp(masses, pos[0])
    cdef np.ndarray[double, ndim=1] next_accelerations
    for j in range(iterations - 1):
        pos[j+1] = pos[j] + vel[j] * dt + 0.5 * cur_accelerations * dt * dt
        next_accelerations = accelerations_notm_nomp(masses, pos[j+1])
        vel[j+1] = vel[j] + 0.5 * (cur_accelerations + next_accelerations) * dt
        cur_accelerations = next_accelerations

    return np.concatenate((pos, vel), 1), times


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] accelerations_notm_mp(np.ndarray[double, ndim=1] masses,
                                                      np.ndarray[double, ndim=1] positions):
    cdef:
        int n_bodies = masses.size
        int dimension = <int>positions.size / <int>masses.size
        double norm3
        np.ndarray[double, ndim=1] res = np.zeros(positions.size)
        np.ndarray[double, ndim=3] tmp = np.zeros((n_bodies, n_bodies, dimension))
        np.ndarray[double, ndim=2] norm = np.zeros((n_bodies, n_bodies))
        double dx
        int i, j, d

    for i in prange(n_bodies, nogil=True, schedule='static'):
        for j in range(n_bodies):
            if i != j:
                for d in range(dimension):
                    dx = positions[j*dimension + d] - positions[i*dimension + d]
                    norm[i, j] += dx * dx
                    tmp[i, j, d] += G*masses[j] * dx
                norm3 = pow(norm[i, j], 1.5)
                for d in range(dimension):
                    res[i*dimension + d] += tmp[i, j, d] / norm3
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def SolveNbodiesVerletCython_notm_mp(np.ndarray[double, ndim=1] masses,
                                     np.ndarray[double, ndim=1] init_pos,
                                     np.ndarray[double, ndim=1] init_vel,
                                     double dt,
                                     int iterations):
    cdef:
        np.ndarray[double, ndim=1] times = np.arange(iterations) * dt
        int half = init_pos.size
        np.ndarray[double, ndim=2] pos = np.empty((times.size, half))
        np.ndarray[double, ndim=2] vel = np.empty((times.size, half))
        int j, d, n
        int dimension = init_pos.shape[0] / masses.shape[0]
        int n_bodies = masses.shape[0]

    pos[0] = init_pos
    vel[0] = init_vel

    cdef double[:] cur_accelerations = accelerations_tm_mp(masses, pos[0])
    cdef double[:] next_accelerations
    for j in range(iterations - 1):
        for n in prange(n_bodies, nogil=True, schedule='static'):
            for d in range(dimension):
                pos[j+1, n*dimension + d] = \
                    pos[j, n*dimension + d]\
                    + vel[j, n*dimension + d] * dt + 0.5 * cur_accelerations[n*dimension + d] * dt * dt

        next_accelerations = accelerations_notm_mp(masses, pos[j+1])

        for n in prange(n_bodies, nogil=True, schedule='static'):
            for d in range(dimension):
                vel[j+1, n*dimension + d] = \
                    vel[j, n*dimension + d] \
                    + 0.5 * (cur_accelerations[n*dimension + d] + next_accelerations[n*dimension + d]) * dt

        cur_accelerations = next_accelerations

    return np.concatenate((pos, vel), 1), times


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] accelerations_tm_nomp(double[:] masses,
                                     double[:] positions):
    cdef:
        int n_bodies = masses.size
        int dimension = <int>positions.size / <int>masses.size
        double norm
        double[:] res = np.zeros(positions.size)
        double[:] tmp = np.zeros(dimension)
        double dx
        int i, j, d

    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                norm = 0
                for d in range(dimension):
                    dx = positions[j*dimension + d] - positions[i*dimension + d]
                    norm += dx * dx
                    tmp[d] = G*masses[j] * dx
                norm = pow(norm, 1.5)
                for d in range(dimension):
                    res[i*dimension + d] += tmp[d] / norm
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def SolveNbodiesVerletCython_tm_nomp(double[:] masses,
                                     double[:] init_pos,
                                     double[:] init_vel,
                                     double dt,
                                     int iterations):
    cdef:
        double[:] times = np.arange(iterations) * dt
        int half = init_pos.size
        double[:,:] pos = np.empty((times.size, half))
        double[:,:] vel = np.empty((times.size, half))
        int j, d, n
        int dimension = init_pos.shape[0] / masses.shape[0]
        int n_bodies = masses.shape[0]

    pos[0] = init_pos
    vel[0] = init_vel

    cdef double[:] cur_accelerations = accelerations_tm_nomp(masses, pos[0])
    cdef double[:] next_accelerations
    for j in range(iterations - 1):
        for d in range(dimension):
            for n in range(n_bodies):
                pos[j+1, n*dimension + d] = \
                    pos[j, n*dimension + d]\
                    + vel[j, n*dimension + d] * dt + 0.5 * cur_accelerations[n*dimension + d] * dt * dt
        next_accelerations = accelerations_tm_nomp(masses, pos[j+1])
        for d in range(dimension):
            for n in range(n_bodies):
                vel[j+1, n*dimension + d] = \
                    vel[j, n*dimension + d] \
                    + 0.5 * (cur_accelerations[n*dimension + d] + next_accelerations[n*dimension + d]) * dt
        cur_accelerations = next_accelerations

    return np.concatenate((pos, vel), 1), times


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] accelerations_tm_mp(double[:] masses,
                                   double[:] positions):
    cdef:
        int n_bodies = masses.size
        int dimension = <int>positions.size / <int>masses.size
        double norm3
        double[:] res = np.zeros(positions.size)
        double[:, :, :] tmp = np.zeros((n_bodies, n_bodies, dimension))
        double[:, :] norm = np.zeros((n_bodies, n_bodies))
        double dx
        int i, j, d

    for i in prange(n_bodies, nogil=True, schedule='static'):
        for j in range(n_bodies):
            if i != j:
                for d in range(dimension):
                    dx = positions[j*dimension + d] - positions[i*dimension + d]
                    norm[i, j] += dx * dx
                    tmp[i, j, d] += G*masses[j] * dx
                norm3 = pow(norm[i, j], 1.5)
                for d in range(dimension):
                    res[i*dimension + d] += tmp[i, j, d] / norm3
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def SolveNbodiesVerletCython_tm_mp(double[:] masses,
                                     double[:] init_pos,
                                     double[:] init_vel,
                                     double dt,
                                     int iterations):
    cdef:
        double[:] times = np.arange(iterations) * dt
        int half = init_pos.size
        double[:,:] pos = np.empty((times.size, half))
        double[:,:] vel = np.empty((times.size, half))
        int j, d, n
        int dimension = init_pos.shape[0] / masses.shape[0]
        int n_bodies = masses.shape[0]

    pos[0] = init_pos
    vel[0] = init_vel

    cdef double[:] cur_accelerations = accelerations_tm_mp(masses, pos[0])
    cdef double[:] next_accelerations
    for j in range(iterations - 1):
        for n in prange(n_bodies, nogil=True, schedule='static'):
            for d in range(dimension):
                pos[j+1, n*dimension + d] = \
                    pos[j, n*dimension + d]\
                    + vel[j, n*dimension + d] * dt + 0.5 * cur_accelerations[n*dimension + d] * dt * dt

        next_accelerations = accelerations_tm_mp(masses, pos[j+1])

        for n in prange(n_bodies, nogil=True, schedule='static'):
            for d in range(dimension):
                vel[j+1, n*dimension + d] = \
                    vel[j, n*dimension + d] \
                    + 0.5 * (cur_accelerations[n*dimension + d] + next_accelerations[n*dimension + d]) * dt

        cur_accelerations = next_accelerations

    return np.concatenate((pos, vel), 1), times