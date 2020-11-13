import numpy as np

import craig


def test_1():
    n = 10
    X = np.random.rand(n, n)
    D = X * np.transpose(X)

    # F = FacilityLocation(D, range(0, n))
    # sset = lazy_greedy(F, range(0, n), 15)

    V = list(range(0, n))
    F = craig.FacilityLocation(D, V)
    sset = craig.lazy_greedy_heap(F, V, 15)

    print(sset)


test_1()
