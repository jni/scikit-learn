cimport cython

import numpy as np
cimport numpy as cnp
cnp.import_array()

cpdef int count_ones(cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] a):
    cdef int i
    cdef int count
    for i in range(len(a)):
        if a[i] == 1: count += 1
    return count

cpdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] continuous_confusion(
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] sorted_y_true,
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] sorted_probas_pred):
    """Return a confusion matrix for every threshold in the prediction."""
    cdef int i
    cdef int curr_threshold = 0
    cdef int num_samples = sorted_y_true.shape[0]
    cdef int it = 0, itp = 1, itn = 2, ifp = 3, ifn = 4
    cdef cnp.float64_t pr = sorted_probas_pred[0], tp = 0, \
        tn = (sorted_y_true != 1).astype(np.float64).sum(), \
        fp = 0, fn = float(num_samples) - tn
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] confusion
    confusion = np.zeros((len(sorted_y_true), 5), dtype=np.float64)
    for i in range(num_samples):
        pr = sorted_probas_pred[i]
        if sorted_y_true[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        confusion[i, 0] = pr
        confusion[i, 1] = tp
        confusion[i, 2] = tn
        confusion[i, 3] = fp
        confusion[i, 4] = fn
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] thresholds
    thresholds, idxs = np.unique(sorted_probas_pred, True)
    return confusion[idxs]
