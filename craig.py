import heapq
import math
import time

import numpy as np


class FacilityLocation:
    def __init__(self, D, V, alpha=1.0):
        """
        Args
        - D: np.array, shape [N, N], similarity matrix
        - V: list of int, indices of columns of D
        - alpha: float
        """
        self.D = D
        self.curVal = 0
        self.curMax = np.zeros(len(D))
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1.0 / self.inc(V, [])

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            if not ndx:  # normalization
                return math.log(1 + self.alpha * 1)
            return (
                self.norm
                * math.log(
                    1
                    + self.f_norm
                    * np.maximum(self.curMax, self.D[:, ndx]).sum()
                )
                - self.curVal
            )
        else:
            return (
                self.norm * math.log(1 + self.f_norm * self.D[:, ndx].sum())
                - self.curVal
            )

    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curMax = np.maximum(self.curMax, self.D[:, ndx])
        else:
            self.curMax = self.D[:, ndx]
        self.curVal = self.norm * math.log(1 + self.f_norm * self.curMax.sum())
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap) - 1)


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def lazy_greedy_heap(F, V, B):
    curVal = 0
    sset = []
    vals = []

    order = []
    heapq._heapify_max(order)
    [_heappush_max(order, (F.inc(sset, index), index)) for index in V]

    while order and len(sset) < B:
        el = _heappop_max(order)
        improv = F.inc(sset, el[1])

        # check for uniques elements
        if improv >= 0:
            if not order:
                curVal = F.add(sset, el[1])
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = _heappop_max(order)
                if improv >= top[0]:
                    curVal = F.add(sset, el[1])
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    _heappush_max(order, (improv, el[1]))
                _heappush_max(order, top)

    return sset, vals


def get_craig_subset_and_weights(similarity_matrix, target_size):
    N = np.shape(similarity_matrix)[0]
    V = list(range(N))

    time_start = time.time()
    subset, _ = lazy_greedy_heap(
        F=FacilityLocation(D=similarity_matrix, V=V), V=V, B=target_size
    )
    total_time = time.time() - time_start

    subset_weights = np.zeros(target_size, dtype=np.float64)
    for i in V:
        subset_weights[np.argmax(similarity_matrix[i, subset])] += 1

    return subset, subset_weights, total_time
