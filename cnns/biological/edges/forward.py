import os
import sys
import struct
import numpy as np



from keras.models import model_from_json



from biologicalgraphs.utilities import dataIO
from biologicalgraphs.utilities.constants import *
from biologicalgraphs.cnns.biological.util import AugmentFeature
from biologicalgraphs.evaluation.classification import Prob2Pred, PrecisionAndRecall



# generator for the inference of the neural network
def EdgeGenerator(examples, width):
    index = 0

    while True:
        # prevent overflow of the queue (these examples will not go through)
        if index == examples.shape[0]: index = 0

        # augment the feature
        example = AugmentFeature(examples[index], width)

        # update the index
        index += 1
        
        yield example



def CollectExamples(prefix, width, radius, subset):
    # get the parent directory with all of the featuers
    parent_directory = 'features/biological/edges-{}nm-{}x{}x{}'.format(radius, width[IB_Z+1], width[IB_Y+1], width[IB_X+1])
    
    positive_filename = '{}/{}/positives/{}-examples.h5'.format(parent_directory, subset, prefix)
    if os.path.exists(positive_filename):
        positive_examples = dataIO.ReadH5File(positive_filename, 'main')
    else:
        positive_examples = np.zeros((0, width[IB_Z + 1], width[IB_Y + 1], width[IB_X + 1]))

    negative_filename = '{}/{}/negatives/{}-examples.h5'.format(parent_directory, subset, prefix)
    if os.path.exists(negative_filename):
        negative_examples = dataIO.ReadH5File(negative_filename, 'main')
    else:
        negative_examples = np.zeros((0, width[IB_Z + 1], width[IB_Y + 1], width[IB_X + 1]))
    
    unknowns_filename = '{}/{}/unknowns/{}-examples.h5'.format(parent_directory, subset, prefix)
    if os.path.exists(unknowns_filename):
        unknowns_example = dataIO.ReadH5File(unknowns_filename, 'main')
    else:
        unknowns_example = np.zeros((0, width[IB_Z + 1], width[IB_Y + 1], width[IB_X + 1]))

    # concatenate all of the examples together
    examples = np.concatenate((positive_examples, negative_examples, unknowns_example), axis=0)
    
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



def CollectEdges(prefix, width, radius, subset):
    # get the parent directory with all of the features
    parent_directory = 'features/biological/edges-{}nm-{}x{}x{}'.format(radius, width[IB_Z+1], width[IB_Y+1], width[IB_X+1])

    examples = []

    positive_filename = '{}/{}/positives/{}.examples'.format(parent_directory, subset, prefix)
    if os.path.exists(positive_filename):
        with open(positive_filename, 'rb') as fd:
            nexamples, = struct.unpack('q', fd.read(8))
            for _ in range(nexamples):
                _, _, _, label_one, label_two, _ = struct.unpack('qqqqqq', fd.read(48))
                examples.append((label_one, label_two))

    negative_filename = '{}/{}/negatives/{}.examples'.format(parent_directory, subset, prefix)
    if os.path.exists(negative_filename):
        with open(negative_filename, 'rb') as fd:
            nexamples, = struct.unpack('q', fd.read(8))
            for _ in range(nexamples):
                _, _, _, label_one, label_two, _ = struct.unpack('qqqqqq', fd.read(48))
                examples.append((label_one, label_two))

    unknowns_filename = '{}/{}/unknowns/{}.examples'.format(parent_directory, subset, prefix)
    if os.path.exists(unknowns_filename):
        with open(unknowns_filename, 'rb') as fd:
            nexamples, = struct.unpack('q', fd.read(8))
            for _ in range(nexamples):
                _, _, _, label_one, label_two, _ = struct.unpack('qqqqqq', fd.read(48))
                examples.append((label_one, label_two))


    # add in information needed for forward inference [regions masked out for training and validation]
    forward_positive_filename = '{}/forward/positives/{}.examples'.format(parent_directory, prefix)
    if os.path.exists(forward_positive_filename):
        with open(forward_positive_filename, 'rb') as fd:
            nexamples, = struct.unpack('q', fd.read(8))
            for _ in range(nexamples):
                _, _, _, label_one, label_two, _ = struct.unpack('qqqqqq', fd.read(48))
                examples.append((label_one, label_two))

    forward_negative_filename = '{}/forward/negatives/{}.examples'.format(parent_directory, prefix)
    if os.path.exists(forward_negative_filename):
        with open(forward_negative_filename, 'rb') as fd:
            nexamples, = struct.unpack('q', fd.read(8))
            for _ in range(nexamples):
                _, _, _, label_one, label_two, _ = struct.unpack('qqqqqq', fd.read(48))
                examples.append((label_one, label_two))

    forward_unknowns_filename = '{}/forward/unknowns/{}.examples'.format(parent_directory, prefix)
    if os.path.exists(forward_unknowns_filename):
        with open(forward_unknowns_filename, 'rb') as fd:
            nexamples, = struct.unpack('q', fd.read(8))
            for _ in range(nexamples):
                _, _, _, label_one, label_two, _ = struct.unpack('qqqqqq', fd.read(48))
                examples.append((label_one, label_two))

    return examples



def Forward(prefix, model_prefix, subset):
    # parameters for the network
    radius = 600
    # there are 3 input channels
    width = (3, 18, 52, 52)

    # read in the trained model
    model = model_from_json(open('{}.json'.format(model_prefix), 'r').read())
    model.load_weights('{}-best-loss.h5'.format(model_prefix))

    # get all of the examples
    examples, npositives, nnegatives = CollectExamples(prefix, width, radius, subset)

    # get the correspond edges
    edges = CollectEdges(prefix, width, radius, subset)
    assert (len(edges) == examples.shape[0])
    
    # get all of the probabilities 
    probabilities = model.predict_generator(EdgeGenerator(examples, width), examples.shape[0], max_q_size=1000)

    # create the correct labels for the ground truth
    ground_truth = np.zeros(npositives + nnegatives, dtype=np.bool)
    for iv in range(npositives):
        ground_truth[iv] = True

    # get the results with labeled data
    predictions = Prob2Pred(np.squeeze(probabilities[:npositives+nnegatives]))

    # print the confusion matrix
    if len(ground_truth):
        output_filename = '{}-{}-inference.txt'.format(model_prefix, prefix)
        PrecisionAndRecall(ground_truth, predictions, output_filename)

    # save the probabilities for each edge
    output_filename = '{}-{}.probabilities'.format(model_prefix, prefix)
    with open(output_filename, 'wb') as fd:
        fd.write(struct.pack('q', examples.shape[0]))
        for ie, (label_one, label_two) in enumerate(edges):
            fd.write(struct.pack('qqd', label_one, label_two, probabilities[ie]))
