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
from callbacks        import IncEpochsFileCallback, tb_logger
from config           import config

ex = Experiment('jhu')

# connect the experiment to the imported modules
IncEpochsFileCallback.on_epoch_end = \
    ex.capture(IncEpochsFileCallback.on_epoch_end)
tb_logger = ex.capture(tb_logger)
config = ex.config(config)

@ex.capture
def save(model, savefile):
    model.save(savefile)

@ex.capture
def train(model, data, labels, epochs_elapsed, epochs, batch_size, n_test):
    tb_log = tb_logger()

    p_test = n_test / len(data)

    model.fit( \
            data, \
            labels, \
            batch_size=batch_size, \
            epochs=epochs_elapsed+epochs, \
            initial_epoch=epochs_elapsed, \
            callbacks=[tb_log, IncEpochsFileCallback()], \
            validation_split=p_test)

@ex.automain
def run(savefile, lr):
    model = None
    try:
        model = load_model(savefile)
        print('Loaded model from', savefile)
    except OSError:
        print('Saved model not found. Making new model...')
        from model import model

    optimizer = SGD(lr=lr, clipnorm=1., decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    train(model)
    save(model)
