import numpy as np

from biologicalgraphs.algorithms.util import CollapseGraph, PrintResults, ReadCandidates


def GraphBaseline(prefix, segmentation, model_prefix, beta=0.95):
    vertex_ones, vertex_twos, edge_weights = ReadCandidates(prefix, model_prefix)

    maintained_edges = np.zeros(edge_weights.size, dtype=np.uint8)

    for ie, edge_weight in enumerate(edge_weights):
        if edge_weight < beta: maintained_edges[ie] = True
        else: maintained_edges[ie] = False

    PrintResults(prefix, vertex_ones, vertex_twos, edge_weights, maintained_edges, 'graph-baseline-{}'.format(int(100 * beta)))

    # create a copy of the segmentation before collapsing
    segmentation = np.copy(segmentation)

    # collapse the graph and save the result
    CollapseGraph(prefix, segmentation, vertex_ones, vertex_twos, maintained_edges, 'graph-baseline-{}'.format(int(100 * beta)))
