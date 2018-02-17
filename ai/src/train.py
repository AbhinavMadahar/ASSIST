#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synopsis: uses the model to determine if an image contains an iceberg or a
# ship.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import numpy as np
import tensorflow as tf
from sacred           import Experiment
from keras.optimizers import SGD
from keras.models     import load_model
from preprocess       import extend
from callbacks        import IncEpochsFileCallback, tb_logger
from config           import config

ex = Experiment('ccore_cnn')

# connect the experiment to the imported modules
IncEpochsFileCallback.on_epoch_end = \
    ex.capture(IncEpochsFileCallback.on_epoch_end)
tb_logger = ex.capture(tb_logger)
config = ex.config(config)

@ex.capture
def save(model, savefile):
    model.save(savefile)

@ex.capture
def train(model, data, labels, epochs_elapsed, epochs, batch_size, n_test, nb_train):
    # generate validation data using the generator
    generator = extend(data, labels)
    validation_data = [generator.__next__() for _ in range(n_test)]
    test_data, test_labels = zip(*validation_data)
    test_data = np.asarray([datum[0] for datum in test_data])
    test_labels = np.asarray(labels[:len(test_data)])
    assert len(test_data) == len(test_labels)

    tb_log = tb_logger()
    model.fit_generator( \
            extend(data, labels), \
            steps_per_epoch=nb_train, \
            epochs=epochs_elapsed+epochs, \
            initial_epoch=epochs_elapsed, \
            verbose=1, \
            callbacks=[tb_log, IncEpochsFileCallback()], \
            validation_data=[test_data, test_labels])

@ex.automain
def run(savefile, lr):
    model = None
    try:
        model = load_model(savefile)
        print('Loaded model from', savefile)
    except OSError:
        print('Saved model not found. Making new model...')
        from model import model

    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr, \
        clipnorm=1.), metrics=['accuracy'])

    train(model)
    save(model)
