#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synopsis: converts the {train,test}.json files to a vector list
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import csv
from collections import namedtuple

Patient = namedtuple('Patient', ['gender', 'age'])

# reads the data from the filename and returns the input data and the correct
# labels
def open_data(filename):
    # we first convert the patients into named tuples with their symptoms/etc.
    # then we turn them into vectors to be fed into the neural network

    patients = []

    # generate the named tuples
    # the file is in Latin-1, not UTF-8
    with open(filename, 'r', encoding='latin-1') as patients_file:
        # we ignore the first line because it defines the headers, not a
        # patient
        patients_file.__next__() # intentionally ignored

        for patient in csv.reader(patients_file):
            patients.append(Patient(patient[3], patient[4]))

    patient_vectors = []

    for patient in patients:
        patient_vector = (
            int(patient.gender == 'Male'),
            int(patient.age)
            )
        patient_vectors.append(patient_vector)

    return patient_vectors
