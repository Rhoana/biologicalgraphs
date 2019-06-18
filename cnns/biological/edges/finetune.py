import os
import keras
import random

import numpy as np

from ibex.utilities import dataIO
from ibex.utilities.constants import *
from ibex.cnns.biological.util import AugmentFeature
from ibex.cnns.biological.edges.train import EdgeNetwork, PlotLosses, WriteLogFiles



# edge generation function is similar except that it only reads files that have dataset in the name
def EdgeGenerator(parameters, width, radius, subset, dataset):
    # SNEMI3D hack
    if subset == 'validation': validation = True
    else: validation = False
    subset = 'training'

    # get the directories corresponding to this radius and subset
    positive_directory = 'features/biological/edges-{}nm-{}x{}x{}/{}/positives'.format(radius, width[IB_Z + 1], width[IB_Y + 1], width[IB_X + 1], subset)
    negative_directory = 'features/biological/edges-{}nm-{}x{}x{}/{}/negatives'.format(radius, width[IB_Z + 1], width[IB_Y + 1], width[IB_X + 1], subset)

    # get all the positive candidate filenames
    positive_filenames = os.listdir(positive_directory)
    positive_candidates = []
    for positive_filename in positive_filenames:
        if not all(restriction in positive_filename for restriction in dataset): continue
        if not positive_filename[-3:] == '.h5': continue
        print positive_filename
        positive_candidates.append(dataIO.ReadH5File('{}/{}'.format(positive_directory, positive_filename), 'main'))
    positive_candidates = np.concatenate(positive_candidates, axis=0)

    # get all the negative candidate filenames
    negative_filenames = os.listdir(negative_directory) 
    negative_candidates = []
    for negative_filename in negative_filenames:
        if not all(restriction in negative_filename for restriction in dataset): continue
        if not negative_filename[-3:] == '.h5': continue
        print negative_filename
        negative_candidates.append(dataIO.ReadH5File('{}/{}'.format(negative_directory, negative_filename), 'main'))
    negative_candidates = np.concatenate(negative_candidates, axis=0)

    if validation: 
        positive_candidates = positive_candidates[int(0.7 * positive_candidates.shape[0]):]
        negative_candidates = negative_candidates[int(0.7 * negative_candidates.shape[0]):]
    else:
        positive_candidates = positive_candidates[:int(0.7 * positive_candidates.shape[0])]
        negative_candidates = negative_candidates[:int(0.7 * negative_candidates.shape[0])]

    # create easy access to the numbers of candidates
    npositive_candidates = positive_candidates.shape[0]
    nnegative_candidates = negative_candidates.shape[0]

    batch_size = parameters['batch_size']

    examples = np.zeros((batch_size, width[0], width[IB_Z+1], width[IB_Y+1], width[IB_X+1]), dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.float32)

    positive_order = range(npositive_candidates)
    negative_order = range(nnegative_candidates)

    random.shuffle(positive_order)
    random.shuffle(negative_order)

    positive_index = 0
    negative_index = 0

    while True:
        for iv in range(batch_size / 2):
            positive_candidate = positive_candidates[positive_order[positive_index]]
            negative_candidate = negative_candidates[negative_order[negative_index]]

            examples[2*iv,:,:,:,:] = AugmentFeature(positive_candidate, width)
            labels[2*iv] = True
            examples[2*iv+1,:,:,:,:] = AugmentFeature(negative_candidate, width)
            labels[2*iv+1] = False

            positive_index += 1
            if positive_index == npositive_candidates:
                random.shuffle(positive_order)
                positive_index = 0
            negative_index += 1
            if negative_index == nnegative_candidates:
                random.shuffle(negative_order)
                negative_index = 0

        yield (examples, labels)




def Finetune(parameters, trained_network_prefix, width, radius, dataset):
    # make sure the model prefix does not contain nodes (to prevent overwriting files)
    assert (not 'nodes' in trained_network_prefix)

    assert (dataset[0] == 'SNEMI3D')

    # identify convenient variables
    starting_epoch = parameters['starting_epoch']
    batch_size = parameters['batch_size']
    examples_per_epoch = parameters['examples_per_epoch']
    weights = parameters['weights']

    model = EdgeNetwork(parameters, width)
    model.load_weights('{}-best-loss.h5'.format(trained_network_prefix))

    root_location = trained_network_prefix.rfind('/')
    output_folder = '{}-{}'.format(trained_network_prefix[:root_location], '-'.join(dataset))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_prefix = '{}/edges'.format(output_folder)

    # open up the log file with no buffer
    logfile = '{}.log'.format(model_prefix)

    # write out the network parameters to a file
    WriteLogFiles(model, model_prefix, parameters)

    # create a set of keras callbacks
    callbacks = []
    
    # save the best model seen so far
    best_loss = keras.callbacks.ModelCheckpoint('{}-best-loss.h5'.format(model_prefix), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    callbacks.append(best_loss)
    best_acc = keras.callbacks.ModelCheckpoint('{}-best-acc.h5'.format(model_prefix), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    callbacks.append(best_acc)
    all_models = keras.callbacks.ModelCheckpoint(model_prefix + '-{epoch:03d}.h5', verbose=0, save_best_only=False, save_weights_only=True, period=5)
    callbacks.append(all_models)

    # plot the loss functions
    plot_losses = PlotLosses(model_prefix)
    callbacks.append(plot_losses)

    # save the json file
    json_string = model.to_json()
    open('{}.json'.format(model_prefix), 'w').write(json_string)
    
    if starting_epoch:
        model.load_weights('{}-{:03d}.h5'.format(model_prefix, starting_epoch))
    
    # there are two thousand validation examples per epoch (standardized)
    nvalidation_examples = 2000

    # train the model
    history = model.fit_generator(EdgeGenerator(parameters, width, radius, 'training', dataset), steps_per_epoch=(examples_per_epoch / batch_size), 
        epochs=250, verbose=1, class_weight=weights, callbacks=callbacks, validation_data=EdgeGenerator(parameters, width, radius, 'validation', dataset), 
                                  validation_steps=(nvalidation_examples / batch_size), initial_epoch=starting_epoch)
    
    with open('{}-history.pickle'.format(model_prefix), 'w') as fd:
        pickle.dump(history.history, fd)

    # save the fully trained model
    model.save_weights('{}.h5'.format(model_prefix))
