#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synopsis: defines callbacks to use while training the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
from sacred           import Experiment
from keras.callbacks  import Callback, TensorBoard

class IncEpochsFileCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        epochs_elapsed = epoch

        with open('epochs.txt', 'w+') as epochs_file:
            epochs_file.write(str(epochs_elapsed))

def tb_logger(n_test, log_dir, batch_size):
    # we want to log to tensorboard to analyse the model
    tb_log = TensorBoard( \
            log_dir=log_dir, \
            histogram_freq=1, \
            batch_size=batch_size, \
            write_graph=True, \
            write_grads=True, \
            write_images=True, \
            embeddings_freq=0, \
            embeddings_layer_names=None, \
            embeddings_metadata=None)

    return tb_log
