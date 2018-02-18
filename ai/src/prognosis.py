#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synopsis: this script intakes the patient's symptoms from STDIN and prints
#           their probability of survival for 14 days to STDOUT as a decimal
#           from 0.00 to 1.00
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy
from keras.models import load_model

modelfilename = 'model.h5'
model = load_model(modelfilename)

age = input('age: ')
gender = input('gender: ')
aspirin = input('aspirin: ')
steroids = input('steroids: ')
noncebhaem = input('noncabhaem: ')
ischaemic = input('ischaemic: ')
haemorrhagic = input('haemorrhagic: ')
recurrent = input('recurrent: ')
haemodilution = input('haemodilution: ')
thromb = input('thromb: ')
patient = numpy.array([numpy.array((age, gender, aspirin, steroids, noncebhaem, ischaemic, haemorrhagic, recurrent, haemodilution, thromb))])

prob_survival = model.predict(patient)
print(prob_survival[0][0])
