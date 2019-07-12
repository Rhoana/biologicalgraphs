import math
import time
import os
import struct
import inspect


cimport cython
cimport numpy as np
from libcpp cimport bool
import ctypes
import numpy as np
import skimage.morphology

from biologicalgraphs.utilities import dataIO
from biologicalgraphs.utilities.constants import *
from medial_axis_util import PostProcess


cdef extern from 'cpp-generate_skeletons.h':
    void CppTopologicalThinning(const char *prefix, long skeleton_resolution[3], const char *lookup_table_directory)
    void CppTeaserSkeletonization(const char *prefix, long skeleton_resolution[3], double input_scale, long input_buffer)
    void CppApplyUpsampleOperation(const char *prefix, const char *params, long *input_segmentation, long skeleton_resolution[3], float output_resolution[3], const char *skeleton_algorithm, double astar_exspanion)


# generate skeletons for this volume
def TopologicalThinning(prefix, input_segmentation, skeleton_resolution=(80, 80, 80), astar_expansion=0):
    # everything needs to be long ints to work with c++
    assert (input_segmentation.dtype == np.int64)

    if not os.path.isdir('benchmarks'): os.mkdir('benchmarks')
    if not os.path.isdir('benchmarks/skeleton'): os.mkdir('benchmarks/skeleton')
   
    # convert the numpy arrays to c++
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_skeleton_resolution = np.ascontiguousarray(skeleton_resolution, dtype=ctypes.c_int64)
    lut_directory = os.path.dirname(__file__)

    # call the topological skeleton algorithm
    CppTopologicalThinning(prefix, &(cpp_skeleton_resolution[0]), lut_directory)
    
    # call the upsampling operation
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_input_segmentation = np.ascontiguousarray(input_segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[float, ndim=1, mode='c'] cpp_output_resolution = np.ascontiguousarray(dataIO.Resolution(prefix), dtype=ctypes.c_float)
    params = ""

    CppApplyUpsampleOperation(prefix, params, &(cpp_input_segmentation[0,0,0]), &(cpp_skeleton_resolution[0]), &(cpp_output_resolution[0]), 'thinning', astar_expansion)
        


# use scipy skeletonization for thinning
def MedialAxis(prefix, input_segmentation, skeleton_resolution=(80, 80, 80), astar_expansion=0):
    # everything needs to be long ints to work with c++
    assert (input_segmentation.dtype == np.int64)

    if not os.path.isdir('benchmarks'): os.mkdir('benchmarks')
    if not os.path.isdir('benchmarks/skeleton'): os.mkdir('benchmarks/skeleton')

    # read the downsampled filename
    input_filename = 'benchmarks/skeleton/{}-downsample-{:03d}x{:03d}x{:03d}.bytes'.format(prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z])
    output_filename = 'benchmarks/skeleton/{}-medial-axis-{:03d}x{:03d}x{:03d}-downsample-skeleton.pts'.format(prefix, skeleton_resolution[IB_X], skeleton_resolution[IB_Y], skeleton_resolution[IB_Z])

    with open(input_filename, 'rb') as rfd, open(output_filename, 'wb') as wfd:
        zres, yres, xres, max_label = struct.unpack('qqqq', rfd.read(32))

        running_times = []

        wfd.write(struct.pack('q', zres))
        wfd.write(struct.pack('q', yres))
        wfd.write(struct.pack('q', xres))
        wfd.write(struct.pack('q', max_label))

        # go through all labels
        for label in range(max_label):
            label_time = time.time()
            segmentation = np.zeros((zres, yres, xres), dtype=np.bool)

            # find topological downsampled locations
            nelements, = struct.unpack('q', rfd.read(8))
            for _ in range(nelements):
                iv, = struct.unpack('q', rfd.read(8))

                iz = iv / (xres * yres)
                iy = (iv - iz * xres * yres) / xres
                ix = iv % xres

                segmentation[iz,iy,ix] = 1

            skeleton = PostProcess(skimage.morphology.skeletonize_3d(segmentation))

            nelements = len(skeleton)
            wfd.write(struct.pack('q', nelements))
            for element in skeleton:
                wfd.write(struct.pack('q', element))
            running_times.append(time.time() - label_time)

    # call the upsampling operation
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_skeleton_resolution = np.ascontiguousarray(skeleton_resolution, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_input_segmentation = np.ascontiguousarray(input_segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[float, ndim=1, mode='c'] cpp_output_resolution = np.ascontiguousarray(dataIO.Resolution(prefix), dtype=ctypes.c_float)
    params = ""

    CppApplyUpsampleOperation(prefix, params, &(cpp_input_segmentation[0,0,0]), &(cpp_skeleton_resolution[0]), &(cpp_output_resolution[0]), 'medial-axis', astar_expansion)



# use TEASER algorithm to generate skeletons
def TEASER(prefix, input_segmentation, skeleton_resolution=(80, 80, 80), teaser_scale=1.3, teaser_buffer=2, astar_expansion=0):
    # everything needs to be long ints to work with c++
    assert (input_segmentation.dtype == np.int64)

    if not os.path.isdir('benchmarks'): os.mkdir('benchmarks')
    if not os.path.isdir('benchmarks/skeleton'): os.mkdir('benchmarks/skeleton')

    # convert to numpy array for c++ call
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_skeleton_resolution = np.ascontiguousarray(skeleton_resolution, dtype=ctypes.c_int64)

    # call the teaser skeletonization algorithm
    CppTeaserSkeletonization(prefix, &(cpp_skeleton_resolution[0]), teaser_scale, teaser_buffer)

    # call the upsampling operation
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_input_segmentation = np.ascontiguousarray(input_segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[float, ndim=1, mode='c'] cpp_output_resolution = np.ascontiguousarray(dataIO.Resolution(prefix), dtype=ctypes.c_float)
    params = "{:02d}-{:02d}".format(long(10 * teaser_scale), teaser_buffer)

    CppApplyUpsampleOperation(prefix, params, &(cpp_input_segmentation[0,0,0]), &(cpp_skeleton_resolution[0]), &(cpp_output_resolution[0]), 'teaser', astar_expansion)