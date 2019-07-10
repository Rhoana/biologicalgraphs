cimport cython
cimport numpy as np

import numpy as np
import ctypes

from biologicalgraphs.algorithms.util import PrintResults, ReadCandidates, CollapseGraph



# c++ external definition
cdef extern from 'cpp-multicut.h':
    unsigned char *CppMulticut(long nvertices, long nedges, long *vertex_ones, long *vertex_twos, double *edge_weights, double beta)



def Multicut(prefix, segmentation, model_prefix, seg2gold_mapping=None):
    # parameter for over/under segmentation
    beta = 0.95

    # get the possible candidates
    vertex_ones, vertex_twos, edge_weights = ReadCandidates(prefix, model_prefix)

    # get the number of vertices and edges
    nvertices = np.amax(segmentation) + 1
    nedges = edge_weights.shape[0]

    # convert to c++ arrays
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_vertex_ones = np.ascontiguousarray(vertex_ones, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_vertex_twos = np.ascontiguousarray(vertex_twos, dtype=ctypes.c_int64)
    cdef np.ndarray[double, ndim=1, mode='c'] cpp_edge_weights = np.ascontiguousarray(edge_weights, dtype=ctypes.c_double)

    # run multicut algorithm to get the edges that should collapse
    cdef unsigned char *cpp_maintained_edges = CppMulticut(nvertices, nedges, &(cpp_vertex_ones[0]), &(cpp_vertex_twos[0]), &(cpp_edge_weights[0]), beta)
    cdef unsigned char[:] tmp_maintained_edges = <unsigned char[:nedges]> cpp_maintained_edges
    maintained_edges = np.asarray(tmp_maintained_edges).astype(dtype=np.bool)

    # output the results
    if not seg2gold_mapping == None:
        PrintResults(prefix, vertex_ones, vertex_twos, edge_weights, maintained_edges, 'multicut-{}'.format(int(100 * beta)))

    # create a copy of the segmentaiton before collapsing
    segmentation = np.copy(segmentation)
    
    # collapse the graph and save the result
    CollapseGraph(prefix, segmentation, vertex_ones, vertex_twos, maintained_edges, 'multicut-{}'.format(int(100 * beta)), not (seg2gold_mapping is None))