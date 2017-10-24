import numpy as np
import random
import threading
import multiprocessing as mp
import time


def matmatSequential(A, B):
    return A.dot(B)

# def matmatThreading(A, B, divide_m, divide_n, divide_k):
#     C = np.empty((A.shape[0], B.shape[1]))
#     for i in range(A.shape[0] / divide_m):
#         for j in range(B.shape[1] / divide_k):
#             for l in range(A.shape[1] / divide_n):


def matmatThreading(A, B):
    def matmatElement(C, i, j):
        C[i, j] = A[i, :].dot(B[:, j])
        return

    C = np.empty((A.shape[0], B.shape[1]))
    processes = []
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            processes.append(threading.Thread(target = matmatElement, args = (C, i, j)))
            processes[-1].start()
    for process in processes:
        process.join()
    return C

def matmatMultiprocessing(A, B):
    def matmatElement(q, rows, cols):
        q.put([A[rows, :].dot(B[:, cols]), rows, cols])

    if __name__ == '__main__':
        q = mp.Queue()
        C = np.empty((A.shape[0], B.shape[1]))
        cpus = mp.cpu_count()
        chunk = C.shape[0] / cpus

        processes = []
        for i in range(cpus):
            processes.append(mp.Process(target = matmatElement, args = (q, np.arange(chunk) + i * chunk, np.arange(C.shape[1]))))
            processes[-1].start()
        # for i in range(C.shape[0]):
        #     for j in range(C.shape[1]):
        #         processes.append(mp.Process(target=matmatElement, args=(q, i, j)))
        #         processes[-1].start()
        for process in processes:
            process.join()
        for i in range(C.size):
            temp = q.get()
            print(temp[0].shape)
            print(temp[1], temp[2])
            # C[temp[1]][temp[2]] = temp[0]
        return C

m = 40
n = 40
k = 40

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
C2 = matmatThreading(A, B)
t2 = time.time()
C3 = matmatMultiprocessing(A, B)
t3 = time.time()
print(t1 - t0, t2- t1, t3 - t2)
print((C1 == C3).all())
