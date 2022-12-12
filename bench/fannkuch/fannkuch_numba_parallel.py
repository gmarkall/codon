# FANNKUCH benchmark
from sys import argv
import time
import numpy as np
import os

os.environ['NUMBA_NUM_THREADS'] = '4'

from numba import njit, prange


@njit
def factorial(x):
    n = 1
    for i in range(2, x+1):
        n *= i
    return n


@njit
def perm(n, i):
    p = np.zeros(n, dtype=np.int64)

    for k in range(n):
        f = factorial(n - 1 - k)
        p[k] = i // f
        i = i % f

    for k in range(n - 1, -1, -1):
        for j in range(k - 1, -1, -1):
            if p[j] <= p[k]:
                p[k] += 1

    return p


@njit(parallel=True)
def fannkuch(n):
    max_flips = 0

    for idx in prange(factorial(n)):
        p = perm(n, idx)
        flips = 0
        k = p[0]

        while k:
            i = 0
            j = k
            while i < j:
                p[i], p[j] = p[j], p[i]
                i += 1
                j -= 1

            k = p[0]
            flips += 1

        max_flips = max(flips, max_flips)

    return max_flips


fannkuch(1)
n = int(argv[1])

t0 = time.perf_counter()
max_flips = fannkuch(n)
t1 = time.perf_counter()

print(f'Pfannkuchen({n}) = {max_flips}')
print(t1 - t0)
