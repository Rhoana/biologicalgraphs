import struct
import numpy as np


from biologicalgraphs.transforms import seg2gold, seg2seg
from biologicalgraphs.data_structures import unionfind
from biologicalgraphs.evaluation.classification import *
from biologicalgraphs.evaluation import comparestacks
from biologicalgraphs.utilities import dataIO



def PrintResults(prefix, vertex_ones, vertex_twos, edge_weights, maintained_edges, algorithm):
    # get the ground truth and print out the results
    seg2gold_mapping = seg2gold.Mapping(prefix)

    # get the number of edges
    nedges = edge_weights.shape[0]

    # see how multicut has changed the results
    labels = []
    cnn_results = []
    multicut_results = []

    # go through each edge
    for ie in range(nedges):
        vertex_one = vertex_ones[ie]
        vertex_two = vertex_twos[ie]

        # skip if there is no ground truth
        if seg2gold_mapping[vertex_one] < 1 or seg2gold_mapping[vertex_two] < 1: continue

        # over 0.5 on edge weight means the edge should collapse
        cnn_results.append(edge_weights[ie] > 0.5)

        # since this edge has ground truth add to list
        # subtract one here since a maintained edge is one that should not be merged
        multicut_results.append(1 - maintained_edges[ie])

        if seg2gold_mapping[vertex_one] == seg2gold_mapping[vertex_two]: labels.append(True)
        else: labels.append(False)

    print 'CNN Results:'
    PrecisionAndRecall(np.array(labels), np.array(cnn_results))

    print 'Multicut Results'
    PrecisionAndRecall(np.array(labels), np.array(multicut_results))



def ReadCandidates(prefix, model_prefix):
    # get the input file with all of the probabilities
    input_filename = '{}-{}.probabilities'.format(model_prefix, prefix)

    # read all of the candidates and probabilities
    with open(input_filename, 'rb') as fd:
        nexamples, = struct.unpack('q', fd.read(8))

        vertex_ones = np.zeros(nexamples, dtype=np.int64)
        vertex_twos = np.zeros(nexamples, dtype=np.int64)
        edge_weights = np.zeros(nexamples, dtype=np.float64)

        for ie in range(nexamples):
            vertex_ones[ie], vertex_twos[ie], edge_weights[ie], = struct.unpack('qqd', fd.read(24))

    # return the list of vertices and corresponding probabilities
    return vertex_ones, vertex_twos, edge_weights



def CollapseGraph(prefix, segmentation, vertex_ones, vertex_twos, maintained_edges, algorithm, evaluate):
    # get the number of edges
    nedges = maintained_edges.shape[0]

    # create the union find data structure and collapse the graph
    max_label = np.amax(segmentation) + 1
    union_find = [unionfind.UnionFindElement(iv) for iv in range(max_label)]

    # go through all of the edges
    for ie in range(nedges):
        # skip if the edge should not collapse
        if maintained_edges[ie]: continue

        # merge these vertices
        vertex_one = vertex_ones[ie]
        vertex_two = vertex_twos[ie]

        unionfind.Union(union_find[vertex_one], union_find[vertex_two])

    # create the mapping and save the result
    mapping = np.zeros(max_label, dtype=np.int64)
    for iv in range(max_label):
        mapping[iv] = unionfind.Find(union_find[iv]).label

    # apply the mapping and save the result
    seg2seg.MapLabels(segmentation, mapping)

    segmentation_filename = 'segmentations/{}-{}.h5'.format(prefix, algorithm)
    dataIO.WriteH5File(segmentation, segmentation_filename, 'main')

    # spawn a new meta file
    dataIO.SpawnMetaFile(prefix, segmentation_filename, 'main')

    # evaluate if gold data exists
    if evaluate:
        # get the variation of information for this result
        new_prefix = segmentation_filename.split('/')[1][:-3]

        # read in the new gold data
        gold = dataIO.ReadGoldData(prefix)

        rand_error, vi = comparestacks.VariationOfInformation(new_prefix, segmentation, gold)

        print 'Rand Error Full: {}'.format(rand_error[0] + rand_error[1])
        print 'Rand Error Merge: {}'.format(rand_error[0])
        print 'Rand Error Split: {}'.format(rand_error[1])

        print 'Variation of Information Full: {}'.format(vi[0] + vi[1])
        print 'Variation of Information Merge: {}'.format(vi[0])
        print 'Variation of Information Split: {}'.format(vi[1])

        with open('results/{}-{}.txt'.format(prefix, algorithm), 'w') as fd:
            fd.write('Rand Error Full: {}\n'.format(rand_error[0] + rand_error[1]))
            fd.write('Rand Error Merge: {}\n'.format(rand_error[0]))
            fd.write('Rand Error Split: {}\n'.format(rand_error[1]))

            fd.write('Variation of Information Full: {}\n'.format(vi[0] + vi[1]))
            fd.write('Variation of Information Merge: {}\n'.format(vi[0]))
            fd.write('Variation of Information Split: {}\n'.format(vi[1]))