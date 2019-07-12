cimport cython
cimport numpy as np

import ctypes
from libcpp cimport bool
import numpy as np
import os



from biologicalgraphs.utilities import dataIO



cdef extern from 'cpp-seg2seg.h':
    void CppDownsampleMapping(const char *prefix, long *segmentation, float input_resolution[3], long output_resolution[3], long input_grid_size[3])
   


def DownsampleMapping(prefix, segmentation, output_resolution=(80, 80, 80)):
    # everything needs to be long ints to work with c++
    assert (segmentation.dtype == np.int64)

    if not os.path.isdir('benchmarks'): os.mkdir('benchmarks')
    if not os.path.isdir('benchmarks/skeleton'): os.mkdir('benchmarks/skeleton')

    # convert numpy arrays to c++ format
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[float, ndim=1, mode='c'] cpp_input_resolution = np.ascontiguousarray(dataIO.Resolution(prefix), dtype=ctypes.c_float)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_output_resolution = np.ascontiguousarray(output_resolution, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_input_grid_size = np.ascontiguousarray(segmentation.shape, dtype=ctypes.c_int64)

    # call c++ function
    CppDownsampleMapping(prefix, &(cpp_segmentation[0,0,0]), &(cpp_input_resolution[0]), &(cpp_output_resolution[0]), &(cpp_input_grid_size[0]))

    # free memory
    del cpp_segmentation
    del cpp_input_resolution
    del cpp_output_resolution
    del cpp_input_grid_size