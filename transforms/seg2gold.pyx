cimport cython
cimport numpy as np

import os
import struct
import numpy as np
import ctypes

from biologicalgraphs.utilities import dataIO



cdef extern from 'cpp-seg2gold.h':
    long *CppMapping(long *segmentation, long *gold, long nentries, double match_threshold, double nonzero_threshold)



def CachedSeg2GoldMapping(prefix):
    # make sure the cache exists
    seg2gold_filename = 'cache/{}-seg2gold.map'.format(prefix)
    assert (os.path.isfile(seg2gold_filename))

    with open(seg2gold_filename, 'rb') as fd:
        max_label, = struct.unpack('q', fd.read(8))

        seg2gold_mapping = np.zeros(max_label, dtype=np.int64)
        for label in range(max_label):
            seg2gold_mapping[label], = struct.unpack('q', fd.read(8))

    return seg2gold_mapping



def Mapping(prefix, segmentation=None, gold=None, match_threshold=0.80, nonzero_threshold=0.40):
    # see if the cache exists and use it if it does
    seg2gold_filename = 'cache/{}-seg2gold.map'.format(prefix)
    if os.path.isfile(seg2gold_filename): 
        return CachedSeg2GoldMapping(prefix)

    if segmentation is None:
        segmentation = dataIO.ReadSegmentationData(prefix)
    if gold is None:
        gold = dataIO.ReadGoldData(prefix)

    # everything needs to be long ints to work with c++
    assert (segmentation.dtype == np.int64)
    assert (gold.dtype == np.int64)

    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_gold = np.ascontiguousarray(gold, dtype=ctypes.c_int64)
    max_segmentation = np.amax(segmentation) + 1

    cdef long *mapping = CppMapping(&(cpp_segmentation[0,0,0]), &(cpp_gold[0,0,0]), segmentation.size, match_threshold, nonzero_threshold)

    cdef long[:] tmp_mapping = <long[:max_segmentation]> mapping;
    seg2gold_mapping = np.asarray(tmp_mapping)

    

    # create cache
    if not os.path.exists('cache'): os.mkdir('cache')
    with open(seg2gold_filename, 'wb') as fd:
        max_label = seg2gold_mapping.size
        fd.write(struct.pack('q', max_label))

        for label in range(max_label):
            fd.write(struct.pack('q', seg2gold_mapping[label]))

    return seg2gold_mapping