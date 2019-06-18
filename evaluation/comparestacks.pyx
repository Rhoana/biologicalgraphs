cimport cython
cimport numpy as np
import numpy as np
import ctypes
import scipy.sparse as sparse
    
from ibex.utilities import dataIO
from ibex.transforms import distance, seg2seg



cdef extern from 'cpp-comparestacks.h':
    double *CppEvaluate(long *segmentation, long *gold, long grid_size[3], long *ground_truth_masks, long nmasks)

def adapted_rand(prefix, seg, gt, all_stats=False, dilate_ground_truth=2, filtersize=0):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index 
    (excluding the zero component of the original labels). Adapted 
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """

    # remove all small connected components
    if filtersize > 0:
        seg2seg.RemoveSmallConnectedComponents(seg, filtersize)
        gt = seg2seg.RemoveSmallConnectedComponents(gt, filtersize)

    if dilate_ground_truth > 0:
        distance.DilateGoldData(prefix, gt, dilate_ground_truth)


    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A,:]
    b = p_ij[1:n_labels_A,1:n_labels_B]
    c = p_ij[1:n_labels_A,0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / sumB
    recall = sumAB / sumA

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
       return (are, precision, recall)
    else:
       return are

       

# with the cache we can now safely run the function consecutive times in a row since we will just read the new ground truth
def VariationOfInformation(prefix, segmentation, gold, dilate_ground_truth=2, input_ground_truth_masks=[0], filtersize=0):
    # make sure not to dilate the ground truth for Fib-25 (already taken care of)
    if 'Fib25' in prefix:
        dilate_ground_truth = 0

    # need to copy the data since there are mutable opeartions below
    ground_truth_masks = np.copy(input_ground_truth_masks).astype(np.int64)
    assert (segmentation.dtype == np.int64)
    assert (gold.dtype == np.int64)
    assert (segmentation.shape == gold.shape)


    # remove all small connected components
    if filtersize > 0:
        seg2seg.RemoveSmallConnectedComponents(segmentation, filtersize)
        seg2seg.RemoveSmallConnectedComponents(gold, filtersize)
   
    if dilate_ground_truth > 0:
        distance.DilateGoldData(prefix, gold, dilate_ground_truth)

    # convert to c++ arrays
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_gold = np.ascontiguousarray(gold, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_input_grid_size = np.ascontiguousarray(segmentation.shape, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_ground_truth_masks = np.ascontiguousarray(ground_truth_masks, dtype=ctypes.c_int64)

    cdef double[:] results = <double[:4]>CppEvaluate(&(cpp_segmentation[0,0,0]), &(cpp_gold[0,0,0]), &(cpp_input_grid_size[0]), &(cpp_ground_truth_masks[0]), ground_truth_masks.size)

    rand_error = (results[0], results[1])
    vi = (results[2], results[3])

    del cpp_input_grid_size
    del cpp_segmentation
    del cpp_gold
    del cpp_ground_truth_masks
    del results

    return (rand_error, vi)