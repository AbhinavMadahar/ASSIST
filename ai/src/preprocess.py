#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synopsis: converts the {train,test}.json files to a vector list
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import json
import numpy as np
import random

# reads the data from the filename and returns the input data and the correct
# labels
def data(filename):
    with open(filename, 'r') as jsonfile:
        json_obj = json.load(jsonfile)

        labels = [[v['is_iceberg']] for v in json_obj]

        # we want the each pixel in the image to have a value from 0 to 1
        # representing its brightness, so we scale linearly
        min_comp = -46 # found by examining data
        max_comp =  35 # ditto
        normalize = lambda comp: (comp-min_comp)/(max_comp-min_comp)
        valid_comp = lambda c: np.isfinite(c) and 0 <= c <= 1

        image_vectors  = [normalize(np.array(x['band_1'])) for x in json_obj]
        assert all(all(valid_comp(c) for c in v) for v in image_vectors), \
                'There is at least 1 nonfinite component in image_vectors'

        image_tensors  = [v.reshape((75,75,1)) for v in image_vectors] # n=3
        data = np.array(image_tensors)

        print("Data shape (n, height, width, depth):", data.shape)

        return data, labels

# uses the data and labels to generate infinitely more data and labels
# by slightly modifying the data without modifying the label
# it acts as a generator, not a list
def extend(original_data, labels):
    def mutate(original_datum):
        mask = np.array([[[random.gauss(1, 0.005)] for _ in range(75)] for _ \
            in range(75)])
        assert original_datum.shape == mask.shape, \
                'Shape mismatch: {} != {}' \
                .format(original_datum.shape, mask.shape)
        new_datum = np.multiply(np.copy(original_datum), mask)
        assert new_datum.shape
        assert new_datum.shape == (75, 75, 1), \
                'Invalid shape: {} should be (75, 75, 1)'.format(new_datum.shape)
        return new_datum

    while True:
        i = random.randrange(len(original_data))
        datum, label = original_data[i], labels[i]
        datum_tensor = np.expand_dims(datum, axis=0)
        assert datum_tensor.shape == (1, 75, 75, 1), \
                'Invalid shape: {} should be (1, 75, 75, 1)' \
                .format(datum_tensor.shape)
        label_tensor = np.expand_dims(label, axis=0)
        assert label_tensor.shape == (1, 1), \
                'Invalid shape: {} should be (1, 1)' \
                .format(label_tensor.shape)
        yield datum_tensor, label_tensor
