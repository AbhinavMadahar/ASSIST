#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synopsis: this script intakes the patient's symptoms and returns their
#           probability of 14-day survival as a decimal from 0.00 to 1.00
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy
from keras.models import load_model

modelfilename = 'model.h5'
model = load_model(modelfilename)

def probability_of_survival(age, gender, face, arm, leg, dysphasia, hemianopia, visuospatial, cerebellar, aspirin, carotid, thromb, stroke_14, haem_14, pulm_14):
    patient = numpy.array([numpy.array((age, gender, face, arm, leg, dysphasia, hemianopia, visuospatial, cerebellar, aspirin, carotid, thromb, stroke_14, haem_14, pulm_14))])
    prob_survival = model.predict(patient)
    return prob_survival[0][0]
