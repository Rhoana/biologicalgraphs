from ibex.transforms import seg2seg, seg2gold
from ibex.data_structures import UnionFind
from ibex.utilities import dataIO
import numpy as np
import os
import struct



def Agglomerate(prefix, model_prefix, threshold=0.5):
    # read the segmentation data 
    segmentation = dataIO.ReadSegmentationData(prefix)

    # get the multicut filename (with graph weights)
    multicut_filename = 'multicut/{}-{}.graph'.format(model_prefix, prefix)

    # get the maximum segmentation value
    max_value = np.amax(segmentation) + 1

    # create union find data structure
    union_find = [UnionFind.UnionFindElement(iv) for iv in range(max_value)]

    # read in all of the labels and merge the result
    with open(multicut_filename, 'rb') as fd:
        # read the number of vertices and edges
        nvertices, nedges, = struct.unpack('QQ', fd.read(16))

        # read in all of the edges
        for ie in range(nedges):
            # read in both labels
            label_one, label_two, = struct.unpack('QQ', fd.read(16))

            # skip over the reduced labels
            fd.read(16)

            # read in the edge weight
            edge_weight, = struct.unpack('d', fd.read(8))

            # merge label one and label two in the union find data structure
            if (edge_weight > threshold):
                UnionFind.Union(union_find[label_one], union_find[label_two])

    # create a mapping
    mapping = np.zeros(max_value, dtype=np.int64)

    # update the segmentation
    for iv in range(max_value):
        label = UnionFind.Find(union_find[iv]).label

        mapping[iv] = label

    # update the labels
    agglomerated_segmentation = seg2seg.MapLabels(segmentation, mapping)

    gold_filename = 'gold/{}_gold.h5'.format(prefix)

    # TODO fix this code temporary filename
    agglomeration_filename = 'multicut/{}-agglomerate.h5'.format(prefix)

    # temporary - write h5 file
    dataIO.WriteH5File(agglomerated_segmentation, agglomeration_filename, 'stack')

    import time
    start_time = time.time()
    print 'Agglomeration - {}:'.format(threshold)
    # create the command line 
    command = '~/software/PixelPred2Seg/comparestacks --stack1 {} --stackbase {} --dilate1 1 --dilatebase 1 --relabel1 --relabelbase --filtersize 100 --anisotropic'.format(agglomeration_filename, gold_filename)

    # execute the command
    os.system(command)
    print time.time() - start_time



def MergeGroundTruth(prefix, model_prefix):
    # read the segmentation data
    segmentation = dataIO.ReadSegmentationData(prefix)
    
    # get the multicut filename (with graph weights)
    multicut_filename = 'multicut/{}-{}.graph'.format(model_prefix, prefix)

    # read the gold data
    gold = dataIO.ReadGoldData(prefix)

    # read in the segmentation to gold mapping
    mapping = seg2gold.Mapping(segmentation, gold)

    # get the maximum segmentation value
    max_value = np.amax(segmentation) + 1

    # create union find data structure
    union_find = [UnionFind.UnionFindElement(iv) for iv in range(max_value)]

    # read in all of the labels 
    with open(multicut_filename, 'rb') as fd:
        # read the number of vertices and edges
        nvertices, nedges, = struct.unpack('QQ', fd.read(16))

        # read in all of the edges
        for ie in range(nedges):
            # read in the two labels
            label_one, label_two, = struct.unpack('QQ', fd.read(16))

            # skip over the reduced labels and edge weight
            fd.read(24)

            # if the labels are the same and the mapping is non zero
            if mapping[label_one] == mapping[label_two] and mapping[label_one]:
                UnionFind.Union(union_find[label_one], union_find[label_two])

    # create a mapping
    mapping = np.zeros(max_value, dtype=np.int64)

    # update the segmentation 
    for iv in range(max_value):
        label = UnionFind.Find(union_find[iv]).label

        mapping[iv] = label

    merged_segmentation = seg2seg.MapLabels(segmentation, mapping)

    gold_filename = 'gold/{}_gold.h5'.format(prefix)

    # TODO fix this code temporary filename
    truth_filename = 'multicut/{}-truth.h5'.format(prefix)

    # temporary write h5 file
    dataIO.WriteH5File(merged_segmentation, truth_filename, 'stack')

    import time
    start_time = time.time()
    print 'Ground truth: '
    # create the command line 
    command = '~/software/PixelPred2Seg/comparestacks --stack1 {} --stackbase {} --dilate1 1 --dilatebase 1 --relabel1 --relabelbase --filtersize 100 --anisotropic'.format(truth_filename, gold_filename)

    # execute the command
    os.system(command)
    print time.time() - start_time
