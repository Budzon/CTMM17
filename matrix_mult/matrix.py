import numpy as np
import random
import threading
import multiprocessing as mp
import time


def matmatSequential(A, B):
    return A.dot(B)

def matmatThreading(A, B):
    def matmatElement(C, rows):
        C[rows, :] = A[rows, :].dot(B)
        return

    C = np.empty((A.shape[0], B.shape[1]))
    cpus = mp.cpu_count()
    chunk = C.shape[0] // cpus

    processes = []
    for i in range(cpus):
        processes.append(threading.Thread(target = matmatElement, args = (C, np.arange(chunk) + i * chunk)))
        processes[-1].start() 
    for process in processes:
        process.join()
    return C

def matmatMultiprocessing(A, B):
    def matmatElement(q, rows):
        q.put([A[rows, :].dot(B), rows])

    if __name__ == '__main__':
        q = mp.Queue()
        C = np.empty((A.shape[0], B.shape[1]))
        cpus = mp.cpu_count()
        chunk = C.shape[0] // cpus

        processes = []
        for i in range(cpus):
            processes.append(mp.Process(target = matmatElement, args = (q, np.arange(chunk) + i * chunk)))
            processes[-1].start()
        for i in range(cpus):
            temp = q.get()
            C[temp[1],:] = temp[0]
        for process in processes:
            process.join()
        return C

m = 10000
n = 2000
k = 1000

A = np.empty((m, n))
B = np.empty((n, k))

for i in range(m):
    for j in range(n):
        A[i, j] = random.randrange(-1, 1)

for i in range(n):
    for j in range(k):
        B[i, j] = random.randrange(-1, 1)

t0 = time.time()
C1 = matmatSequential(A, B)
t1 = time.time()
print(t1 - t0)
C2 = matmatThreading(A, B)
t2 = time.time()
print(t2 - t1)
C3 = matmatMultiprocessing(A, B)
t3 = time.time()
print(t3 - t2)
