cimport cython
cimport numpy as np
import numpy as np
import ctypes
    


from biologicalgraphs.utilities import dataIO
from biologicalgraphs.transforms import distance, seg2seg



cdef extern from 'cpp-comparestacks.h':
    double *CppEvaluate(long *segmentation, long *gold, long grid_size[3], long *ground_truth_masks, long nmasks)

       

# with the cache we can now safely run the function consecutive times in a row since we will just read the new ground truth
def VariationOfInformation(prefix, segmentation, gold, dilate_ground_truth=2, input_ground_truth_masks=[0], filtersize=0):
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