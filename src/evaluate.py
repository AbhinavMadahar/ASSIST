#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synopsis: evaluates how well the current model can predict 14-day survival
#           using the entire dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from preprocess import open_data
from prognosis import probability_of_survival
from keras.models import load_model

model = load_model('model.h5')

print('Evaluating model...')
n_correct = 0
n_total = 0
for patient, survived in zip(*open_data('../data/patients.csv')):
    p_survival = probability_of_survival(*patient)
    predicted_to_survive = round(p_survival)
    if predicted_to_survive == survived:
        n_correct += 1
    n_total += 1

print('Percentage correctly predicted', n_correct/n_total)
