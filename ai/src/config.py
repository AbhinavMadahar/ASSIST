#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synopsis: configures the experimental setup (NOT the model architecture)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from preprocess import open_data

def config():
    # set the training variables (NONE of these are model variables)
    data, labels = open_data('../data/patients.csv', 1000000)
    epochs = 32
    batch_size = 64
    lr = 0.01
    savefile = 'model.h5'
    log_dir = 'log'
    epochs_elapsed = 0
    n_test = 10000 # number of patients to use for testing the neural network

    try:
        with open('epochs.txt', 'r') as epochs_file:
            epochs_elapsed = int(epochs_file.read().strip())
    except (OSError, ValueError):
        with open('epochs.txt', 'w') as epochs_file:
            epochs_file.write('0')
