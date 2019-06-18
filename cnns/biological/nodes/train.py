import os
import matplotlib
import pickle
matplotlib.use('Agg')
import random

import numpy as np

import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
import keras

from ibex.utilities import dataIO
from ibex.utilities.constants import *
from ibex.cnns.biological.util import AugmentFeature


# add a convolutional layer to the model
def ConvolutionalLayer(model, filter_size, kernel_size, padding, activation, normalization, input_shape=None):
    if not input_shape == None: model.add(Convolution3D(filter_size, kernel_size, padding=padding, input_shape=input_shape))
    else: model.add(Convolution3D(filter_size, kernel_size, padding=padding))

    # add activation layer
    if activation == 'LeakyReLU': model.add(LeakyReLU(alpha=0.001))
    else: model.add(Activation(activation))
    
    # add normalization after activation
    if normalization: model.add(BatchNormalization())



# add a pooling layer to the model
def PoolingLayer(model, pool_size, dropout, normalization):
    model.add(MaxPooling3D(pool_size=pool_size))

    # add normalization before dropout
    if normalization: model.add(BatchNormalization())

    # add dropout layer
    if dropout > 0.0: model.add(Dropout(dropout))



# add a flattening layer to the model
def FlattenLayer(model):
    model.add(Flatten())



# add a dense layer to the model
def DenseLayer(model, filter_size, dropout, activation, normalization):
    model.add(Dense(filter_size))
    if (dropout > 0.0): model.add(Dropout(dropout))

    # add activation layer
    if activation == 'LeakyReLU': model.add(LeakyReLU(alpha=0.001))
    else: model.add(Activation(activation))

    # add normalization after activation
    if normalization: model.add(BatchNormalization())



class PlotLosses(keras.callbacks.Callback):
    def __init__(self, model_prefix):
        super(PlotLosses, self).__init__()
        self.model_prefix = model_prefix

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i = self.i + 1
        
        plt.ylabel('Training/Validation Loss')
        plt.xlabel('Number of Epochs')
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        plt.savefig('{}-training-curve.png'.format(self.model_prefix))
        plt.gcf().clear()



def NodeNetwork(parameters, width):
    # identify convenient variables
    initial_learning_rate = parameters['initial_learning_rate']
    decay_rate = parameters['decay_rate']
    activation = parameters['activation']
    normalization = parameters['normalization']
    filter_sizes = parameters['filter_sizes']
    depth = parameters['depth']
    optimizer = parameters['optimizer']
    betas = parameters['betas']
    loss_function = parameters['loss_function']
    assert (len(filter_sizes) >= depth)

    model = Sequential()

    ConvolutionalLayer(model, filter_sizes[0], (3, 3, 3), 'valid', activation, normalization, width)
    ConvolutionalLayer(model, filter_sizes[0], (3, 3, 3), 'valid', activation, normalization)
    PoolingLayer(model, (1, 2, 2), 0.2, normalization)

    ConvolutionalLayer(model, filter_sizes[1], (3, 3, 3), 'valid', activation, normalization)
    ConvolutionalLayer(model, filter_sizes[1], (3, 3, 3), 'valid', activation, normalization)
    PoolingLayer(model, (1, 2, 2), 0.2, normalization)

    ConvolutionalLayer(model, filter_sizes[2], (3, 3, 3), 'valid', activation, normalization)
    ConvolutionalLayer(model, filter_sizes[2], (3, 3, 3), 'valid', activation, normalization)
    PoolingLayer(model, (2, 2, 2), 0.2, normalization)

    if depth > 3:
        ConvolutionalLayer(model, filter_sizes[3], (3, 3, 3), 'valid', activation, normalization)
        ConvolutionalLayer(model, filter_sizes[3], (3, 3, 3), 'valid', activation, normalization)
        PoolingLayer(model, (2, 2, 2), 0.2, normalization)


    FlattenLayer(model)
    DenseLayer(model, 512, 0.2, activation, normalization)
    DenseLayer(model, 1, 0.5, 'sigmoid', False)

    if optimizer == 'adam': opt = Adam(lr=initial_learning_rate, decay=decay_rate, beta_1=betas[0], beta_2=betas[1], epsilon=1e-08)
    elif optimizer == 'nesterov': opt = SGD(lr=initial_learning_rate, decay=decay_rate, momentum=0.99, nesterov=True)
    model.compile(loss=loss_function, optimizer=opt, metrics=['mean_squared_error', 'accuracy'])
    
    return model



# write all relevant information to the log file
def WriteLogFiles(model, model_prefix, parameters):
    logfile = '{}.log'.format(model_prefix)

    with open(logfile, 'w') as fd:
        for layer in model.layers:
            print '{} {} -> {}'.format(layer.get_config()['name'], layer.input_shape, layer.output_shape)
            fd.write('{} {} -> {}\n'.format(layer.get_config()['name'], layer.input_shape, layer.output_shape))
        print 
        fd.write('\n')
        for parameter in parameters:
            print '{}: {}'.format(parameter, parameters[parameter])
            fd.write('{}: {}\n'.format(parameter, parameters[parameter]))



def NodeGenerator(parameters, width, radius, subset):
    # get the directories corresponding to this radius and subset
    positive_directory = 'features/biological/nodes-{}nm-{}x{}x{}/{}/positives'.format(radius, width[IB_Z + 1], width[IB_Y + 1], width[IB_X + 1], subset)
    negative_directory = 'features/biological/nodes-{}nm-{}x{}x{}/{}/negatives'.format(radius, width[IB_Z + 1], width[IB_Y + 1], width[IB_X + 1], subset)

    # get all the positive candidate filenames
    positive_filenames = os.listdir(positive_directory)
    positive_candidates = []
    for positive_filename in positive_filenames:
        if not 'PNI' in positive_filename: continue
        if not positive_filename[-3:] == '.h5': continue
        positive_candidates.append(dataIO.ReadH5File('{}/{}'.format(positive_directory, positive_filename), 'main'))
    positive_candidates = np.concatenate(positive_candidates, axis=0)

    # get all the negative candidate filenames
    negative_filenames = os.listdir(negative_directory) 
    negative_candidates = []
    for negative_filename in negative_filenames:
        if not 'PNI' in negative_filename: continue
        if not negative_filename[-3:] == '.h5': continue
        negative_candidates.append(dataIO.ReadH5File('{}/{}'.format(negative_directory, negative_filename), 'main'))
    negative_candidates = np.concatenate(negative_candidates, axis=0)

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



def Train(parameters, model_prefix, width, radius):
    # make sure the model prefix does not contain edges (to prevent overwriting files)
    assert (not 'edges' in model_prefix)

    # identify convenient variables
    starting_epoch = parameters['starting_epoch']
    batch_size = parameters['batch_size']
    examples_per_epoch = parameters['examples_per_epoch']
    weights = parameters['weights']

    model = NodeNetwork(parameters, width)

    # make sure the folder for the model prefix exists
    root_location = model_prefix.rfind('/')
    output_folder = model_prefix[:root_location]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
    history = model.fit_generator(NodeGenerator(parameters, width, radius, 'training'), steps_per_epoch=(examples_per_epoch / batch_size), 
        epochs=2000, verbose=1, class_weight=weights, callbacks=callbacks, validation_data=NodeGenerator(parameters, width, radius, 'validation'), 
                                  validation_steps=(nvalidation_examples / batch_size), initial_epoch=starting_epoch)
    
    with open('{}-history.pickle'.format(model_prefix), 'w') as fd:
        pickle.dump(history.history, fd)

    # save the fully trained model
    model.save_weights('{}.h5'.format(model_prefix))
