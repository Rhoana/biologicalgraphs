import os



cimport cython
cimport numpy as np
from libcpp cimport bool
import ctypes
import numpy as np


from biologicalgraphs.utilities import dataIO
from biologicalgraphs.utilities.constants import *



cdef extern from 'cpp-generate_skeletons.h':
    void CppTopologicalThinning(const char *prefix, long skeleton_resolution[3], const char *lookup_table_directory)
    void CppFindEndpointVectors(const char *prefix, long skeleton_resolution[3], float output_resolution[3])
    void CppApplyUpsampleOperation(const char *prefix, long *input_segmentation, long skeleton_resolution[3], float output_resolution[3])



# generate skeletons for this volume
def TopologicalThinning(prefix, input_segmentation):
    # resolution for the skeleton
    skeleton_resolution=(80, 80, 80)
    
    # everything needs to be long ints to work with c++
    assert (input_segmentation.dtype == np.int64)
   
    # convert the numpy arrays to c++
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_skeleton_resolution = np.ascontiguousarray(skeleton_resolution, dtype=ctypes.c_int64)
    lut_directory = os.path.dirname(__file__)

    # call the topological skeleton algorithm
    CppTopologicalThinning(prefix, &(cpp_skeleton_resolution[0]), lut_directory)
    
    # call the upsampling operation
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_input_segmentation = np.ascontiguousarray(input_segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[float, ndim=1, mode='c'] cpp_output_resolution = np.ascontiguousarray(dataIO.Resolution(prefix), dtype=ctypes.c_float)
    
    CppApplyUpsampleOperation(prefix, &(cpp_input_segmentation[0,0,0]), &(cpp_skeleton_resolution[0]), &(cpp_output_resolution[0]))
        


# find endpoint vectors for this skeleton
def FindEndpointVectors(prefix):
    # resolution for the skeletons
    skeleton_resolution=(80, 80, 80)

    # convert to numpy array for c++ call
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_skeleton_resolution = np.ascontiguousarray(skeleton_resolution, dtype=ctypes.c_int64)
    cdef np.ndarray[float, ndim=1, mode='c'] cpp_output_resolution = np.ascontiguousarray(dataIO.Resolution(prefix), dtype=ctypes.c_float)

    CppFindEndpointVectors(prefix, &(cpp_skeleton_resolution[0]), &(cpp_output_resolution[0]))