cimport cython
cimport numpy as np

import os
import ctypes
import numpy as np



from biologicalgraphs.utilities import dataIO



cdef extern from 'cpp-distance.h':
    float *CppTwoDimensionalDistanceTransform(long *data, long grid_size[3])
    void CppDilateGoldData(long *data, long grid_size[3], float distance)



# get the two dimensional distance transform
def TwoDimensionalDistanceTransform(data):
    # convert segmentation to int64
    if not data.dtype == np.int64: data = data.astype(np.int64)

    # convert numpy array to c++
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_data = np.ascontiguousarray(data, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_input_grid_size = np.ascontiguousarray(data.shape, dtype=ctypes.c_int64)

    # get the distance transform
    cdef float[:] distances = <float[:data.size]>CppTwoDimensionalDistanceTransform(&(cpp_data[0,0,0]), &(cpp_input_grid_size[0]))

    del cpp_input_grid_size

    return np.reshape(np.asarray(distances), data.shape)



# dilate segments from boundaries
def DilateGoldData(prefix, data, distance):
    # see if a cache exists for this file
    gold_filename = dataIO.GetGoldFilename(prefix)
    cached_filename = '{}-dilated-{}.h5'.format(gold_filename[:-3], distance)

    # read the cached file and leave the function    
    if os.path.exists(cached_filename):
        dilated_data = dataIO.ReadH5File(cached_filename, 'main')
        np.copyto(data, dilated_data, casting='no')
        return

    # convert segmentation to int64
    if not data.dtype == np.int64: data = data.astype(np.int64)

    # convert numpy array to c++
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_data = np.ascontiguousarray(data, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_input_grid_size = np.ascontiguousarray(data.shape, dtype=ctypes.c_int64)

    # dilate the data by distance
    CppDilateGoldData(&(cpp_data[0,0,0]), &(cpp_input_grid_size[0]), float(distance))

    del cpp_input_grid_size

    # save this data
    dataIO.WriteH5File(data, cached_filename, 'main')