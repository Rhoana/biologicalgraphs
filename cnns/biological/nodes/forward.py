import os
import numpy as np
import sys
import struct
import time

from keras.models import model_from_json

from biologicalgraphs.utilities import dataIO
from biologicalgraphs.utilities.constants import *
from biologicalgraphs.transforms import seg2seg, seg2gold
from biologicalgraphs.cnns.biological.util import AugmentFeature
from biologicalgraphs.evaluation.classification import Prob2Pred, PrecisionAndRecall
from biologicalgraphs.graphs.biological.util import FindSmallSegments
from biologicalgraphs.evaluation import comparestacks



# generator for the inference of the neural network
def NodeGenerator(examples, width):
    index = 0

    start_time = time.time()

    while True:             
        if index and not (index % 1000):
            print '{}/{} in {:0.2f} seconds'.format(index, examples.shape[0], time.time() - start_time)
            start_time = time.time()
        # prevent overflow of the queue (these examples will not go through)
        if index == examples.shape[0]: index = 0

        # augment the feature
        example = AugmentFeature(examples[index], width)

        # update the index
        index += 1
        
        yield example



def CollectExamples(prefix, width, radius, subset):
    # get the parent directory with all of the featuers
    parent_directory = 'features/biological/nodes-{}nm-{}x{}x{}'.format(radius, width[IB_Z+1], width[IB_Y+1], width[IB_X+1])
    
    positive_filename = '{}/{}/positives/{}-examples.h5'.format(parent_directory, subset, prefix)
    positive_examples = dataIO.ReadH5File(positive_filename, 'main')
    
    negative_filename = '{}/{}/negatives/{}-examples.h5'.format(parent_directory, subset, prefix)
    negative_examples = dataIO.ReadH5File(negative_filename, 'main')
    
    unknowns_filename = '{}/{}/unknowns/{}-examples.h5'.format(parent_directory, subset, prefix)
    unknowns_examples = dataIO.ReadH5File(unknowns_filename, 'main')
    
    # concatenate all of the examples together
    examples = np.concatenate((positive_examples, negative_examples, unknowns_examples), axis=0)

    # add in information needed for forward inference [regions masked out for training and validation]
    forward_positive_filename = '{}/forward/positives/{}-examples.h5'.format(parent_directory, prefix)
    if os.path.exists(forward_positive_filename):
        forward_positive_examples = dataIO.ReadH5File(forward_positive_filename, 'main')
        examples = np.concatenate((examples, forward_positive_examples), axis=0)

    forward_negative_filename = '{}/forward/negatives/{}-examples.h5'.format(parent_directory, prefix)
    if os.path.exists(forward_negative_filename):
        forward_negative_examples = dataIO.ReadH5File(forward_negative_filename, 'main')
        examples = np.concatenate((examples, forward_negative_examples), axis=0)

    forward_unknowns_filename = '{}/forward/unknowns/{}-examples.h5'.format(parent_directory, prefix)
    if os.path.exists(forward_unknowns_filename):
        forward_unknowns_examples = dataIO.ReadH5File(forward_unknowns_filename, 'main')
        examples = np.concatenate((examples, forward_unknowns_examples), axis=0)

    return examples, positive_examples.shape[0], negative_examples.shape[0]



def CollectLargeSmallPairs(prefix, width, radius, subset):
    # get the parent directory with all of the featuers
    parent_directory = 'features/biological/nodes-{}nm-{}x{}x{}'.format(radius, width[IB_Z+1], width[IB_Y+1], width[IB_X+1])
    
    examples = []

    positive_filename = '{}/{}/positives/{}.examples'.format(parent_directory, subset, prefix)
    with open(positive_filename, 'rb') as fd:
        nexamples, = struct.unpack('q', fd.read(8))
        for _ in range(nexamples):
            label_one, label_two, = struct.unpack('qq', fd.read(16))
            examples.append((label_one, label_two))

    negative_filename = '{}/{}/negatives/{}.examples'.format(parent_directory, subset, prefix)
    with open(negative_filename, 'rb') as fd:
        nexamples, = struct.unpack('q', fd.read(8))
        for _ in range(nexamples):
            label_one, label_two, = struct.unpack('qq', fd.read(16))
            examples.append((label_one, label_two))

    unknowns_filename = '{}/{}/unknowns/{}.examples'.format(parent_directory, subset, prefix)
    with open(unknowns_filename, 'rb') as fd:
        nexamples, = struct.unpack('q', fd.read(8))
        for _ in range(nexamples):
            label_one, label_two, = struct.unpack('qq', fd.read(16))
            examples.append((label_one, label_two))


    # add in information needed for forward inference [regions masked out for training and validation]
    forward_positive_filename = '{}/forward/positives/{}.examples'.format(parent_directory, prefix)
    if os.path.exists(forward_positive_filename):
        with open(forward_positive_filename, 'rb') as fd:
            nexamples, = struct.unpack('q', fd.read(8))
            for _ in range(nexamples):
                label_one, label_two, = struct.unpack('qq', fd.read(16))
                examples.append((label_one, label_two))

    forward_negative_filename = '{}/forward/negatives/{}.examples'.format(parent_directory, prefix)
    if os.path.exists(forward_negative_filename):
        with open(forward_negative_filename, 'rb') as fd:
            nexamples, = struct.unpack('q', fd.read(8))
            for _ in range(nexamples):
                label_one, label_two, = struct.unpack('qq', fd.read(16))
                examples.append((label_one, label_two))

    forward_unknowns_filename = '{}/forward/unknowns/{}.examples'.format(parent_directory, prefix)
    if os.path.exists(forward_unknowns_filename):
        with open(forward_unknowns_filename, 'rb') as fd:
            nexamples, = struct.unpack('q', fd.read(8))
            for _ in range(nexamples):
                label_one, label_two, = struct.unpack('qq', fd.read(16))
                examples.append((label_one, label_two))

    return examples



def Forward(prefix, model_prefix, segmentation, width, radius, subset, evaluate=False, threshold_volume=10368000):
    # read in the trained model
    model = model_from_json(open('{}.json'.format(model_prefix), 'r').read())
    model.load_weights('{}-best-loss.h5'.format(model_prefix))

    # get all of the examples
    examples, npositives, nnegatives = CollectExamples(prefix, width, radius, subset)
    
    # get all of the large-small pairings
    pairings = CollectLargeSmallPairs(prefix, width, radius, subset)
    #assert (len(pairings) == examples.shape[0])
    
    # get the threshold in terms of number of voxels
    resolution = dataIO.Resolution(prefix)
    threshold = int(threshold_volume / (resolution[IB_Z] * resolution[IB_Y] * resolution[IB_X]))

    # get the list of nodes over and under the threshold
    small_segments, large_segments = FindSmallSegments(segmentation, threshold)
 
    # get all of the probabilities 
    probabilities = model.predict_generator(NodeGenerator(examples, width), examples.shape[0], max_q_size=1000)

    # save the probabilities to a file
    output_filename = '{}-{}.probabilities'.format(model_prefix, prefix)
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('q', examples.shape[0]))
        for ie, (label_one, label_two) in enumerate(pairings):
            fd.write(struct.pack('qqd', label_one, label_two, probabilities[ie]))

    # create the correct labels for the ground truth
    ground_truth = np.zeros(npositives + nnegatives, dtype=np.bool)
    for iv in range(npositives):
        ground_truth[iv] = True
    
    # get the results with labeled data
    predictions = Prob2Pred(np.squeeze(probabilities[:npositives+nnegatives]))
    
    # print the confusion matrix
    output_filename = '{}-{}-inference.txt'.format(model_prefix, prefix)
    PrecisionAndRecall(ground_truth, predictions, output_filename)
    
    # create a mapping 
    small_segment_predictions = dict()
    for small_segment in small_segments:
        small_segment_predictions[small_segment] = set()

    # go through each pairing
    for pairing, probability in zip(pairings, probabilities):
        label_one, label_two = pairing
        # make sure that either label one or two is small and the other is large
        assert ((label_one in small_segments) ^ (label_two in small_segments))

        if label_one in small_segments:
            small_segment = label_one
            large_segment = label_two
        else:
            small_segment = label_two
            large_segment = label_one
            
        small_segment_predictions[small_segment].add((large_segment, probability[0]))

    # begin to map the small labels
    max_label = np.amax(segmentation) + 1
    mapping = [iv for iv in range(max_label)]

    # look at seg2gold to see how many correct segments are merged
    seg2gold_mapping = seg2gold.Mapping(prefix)
    
    ncorrect_merges = 0
    nincorrect_merges = 0

    # go through all of the small segments
    for small_segment in small_segments:
        best_probability = -1
        best_large_segment = -1

        # go through all the neighboring large segments
        for large_segment, probability in small_segment_predictions[small_segment]:
            if probability > best_probability:
                best_probability = probability
                best_large_segment = large_segment
        
        # this should almost never happen but if it does just continue
        if best_large_segment == -1 or best_probability < 0.5:
            mapping[small_segment] = small_segment
            continue
        # get all of the best large segments
        else:
            mapping[small_segment] = best_large_segment

        # don't consider undetermined locations
        if seg2gold_mapping[small_segment] < 1 or seg2gold_mapping[best_large_segment] < 1: continue

        if seg2gold_mapping[small_segment] == seg2gold_mapping[best_large_segment]: ncorrect_merges += 1
        else: nincorrect_merges += 1

    print '\nResults:'
    print '  Correctly Merged: {}'.format(ncorrect_merges)
    print '  Incorrectly Merged: {}'.format(nincorrect_merges)
    
    with open(output_filename, 'a') as fd:
        fd.write('\nResults:\n')
        fd.write('  Correctly Merged: {}\n'.format(ncorrect_merges))
        fd.write('  Incorrectly Merged: {}\n'.format(nincorrect_merges))
    
    # save the node mapping in the cache for later
    end2end_mapping = [mapping[iv] for iv in range(max_label)]

    # initiate the mapping to eliminate small segments
    seg2seg.MapLabels(segmentation, mapping)

    # reduce the labels and map again
    mapping, _ = seg2seg.ReduceLabels(segmentation)
    seg2seg.MapLabels(segmentation, mapping)

    # update the end to end mapping with the reduced labels
    for iv in range(max_label):
        end2end_mapping[iv] = mapping[end2end_mapping[iv]]
    
    # get the model name (first component is architecture and third is node-)
    model_name = model_prefix.split('/')[1]
    output_filename = 'segments/{}-reduced-{}.h5'.format(prefix, model_name)
    dataIO.WriteH5File(segmentation, output_filename, 'main')

    # spawn a new meta file
    dataIO.SpawnMetaFile(prefix, output_filename, 'main')
    
    # save the end to end mapping in the cache
    mapping_filename = 'cache/{}-reduced-{}-end2end.map'.format(prefix, model_name)
    with open(mapping_filename, 'wb') as fd:
        fd.write(struct.pack('q', max_label))
        for label in range(max_label):
            fd.write(struct.pack('q', end2end_mapping[label]))

    if evaluate:
        gold = dataIO.ReadGoldData(prefix)

        # run the evaluation framework
        rand_error, vi = comparestacks.VariationOfInformation(segmentation, gold)

        # write the output file
        with open('node-results/{}-reduced-{}.txt'.format(prefix, model_name), 'w') as fd:
            fd.write('Rand Error Full: {}\n'.format(rand_error[0] + rand_error[1]))
            fd.write('Rand Error Merge: {}\n'.format(rand_error[0]))
            fd.write('Rand Error Split: {}\n'.format(rand_error[1]))

            fd.write('Variation of Information Full: {}\n'.format(vi[0] + vi[1]))
            fd.write('Variation of Information Merge: {}\n'.format(vi[0]))
            fd.write('Variation of Information Split: {}\n'.format(vi[1]))
