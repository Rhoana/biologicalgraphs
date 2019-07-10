cimport cython
cimport numpy as np

import math
import numpy as np
import ctypes
import scipy

from biologicalgraphs.algorithms.util import PrintResults, ReadCandidates, CollapseGraph



# c++ external definition
cdef extern from 'cpp-lifted-multicut.h':
    unsigned char *CppLiftedMulticut(long nvertices, long nedges, long *vertex_ones, long *vertex_twos, double *edge_weights, double beta)




def GenerateLiftedEdges(vertex_ones, vertex_twos, edge_weights, nvertices):
    # get the number of normal edges
    nedges = edge_weights.size

    # up the edge weights to be the negative log likelihood for dijkstra
    negative_log_weights = np.zeros(nedges, dtype=np.float32)
    for ie in range(nedges):
        negative_log_weights[ie] = -math.log(edge_weights[ie])

    # create a sparse graph with these edges
    sparse_graph = scipy.sparse.coo_matrix((negative_log_weights, (vertex_ones, vertex_twos)), shape=(nvertices, nvertices))
    dijkstra_distance = scipy.sparse.csgraph.dijkstra(sparse_graph, directed=False)

    # need to convert back to probabilities from negative log likelihoods
    return np.exp(-1 * dijkstra_distance)




def LiftedMulticut(prefix, segmentation, model_prefix, seg2gold_mapping=None):
    # parameter for over/under segmentation
    beta = 0.95

    # get the possible candidates
    vertex_ones, vertex_twos, edge_weights = ReadCandidates(prefix, model_prefix)

    # get the number of vertices and edges
    nvertices = np.amax(segmentation) + 1
    nedges = edge_weights.shape[0]

    lifted_edge_weights = GenerateLiftedEdges(vertex_ones, vertex_twos, edge_weights, nvertices)

    # convert to c++ arrays
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_vertex_ones = np.ascontiguousarray(vertex_ones, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_vertex_twos = np.ascontiguousarray(vertex_twos, dtype=ctypes.c_int64)
    cdef np.ndarray[double, ndim=2, mode='c'] cpp_lifted_edge_weights = np.ascontiguousarray(lifted_edge_weights, dtype=ctypes.c_double)

    # run multicut algorithm to get the edges that should collapse
    cdef unsigned char *cpp_maintained_edges = CppLiftedMulticut(nvertices, nedges, &(cpp_vertex_ones[0]), &(cpp_vertex_twos[0]), &(cpp_lifted_edge_weights[0,0]), beta)
    cdef unsigned char[:] tmp_maintained_edges = <unsigned char[:nedges]> cpp_maintained_edges
    maintained_edges = np.asarray(tmp_maintained_edges).astype(dtype=np.bool)

    # output the results
    if not seg2gold_mapping is None:
        PrintResults(prefix, vertex_ones, vertex_twos, edge_weights, maintained_edges, 'lifted-multicut-{}'.format(int(100 * beta)))

    # create a copy of the segmentation before collapsing
    segmentation = np.copy(segmentation)
    
    # collapse the graph and save the result
    CollapseGraph(prefix, segmentation, vertex_ones, vertex_twos, maintained_edges, 'lifted-multicut-{}'.format(int(100 * beta)), not (seg2gold_mapping is None))