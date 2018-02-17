#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synopsis: configures the experimental setup (NOT the model architecture)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from preprocess import data

def config():
    # set the training variables (NONE of these are model variables)
    data, labels = data('data/train.json')
    epochs = 64
    batch_size = 64
    lr = 0.01
    savefile = 'model.h5'
    log_dir = 'log'
    epochs_elapsed = 0
    n_test = 128 # how many images to use to test the model
    nb_train = 32 # for how many batches to train per epoch

    try:
        with open('epochs.txt', 'r') as epochs_file:
            epochs_elapsed = int(epochs_file.read().strip())
    except (OSError, ValueError):
        with open('epochs.txt', 'w') as epochs_file:
            epochs_file.write('0')
