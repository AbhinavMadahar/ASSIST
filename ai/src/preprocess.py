#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synopsis: converts the {train,test}.json files to a vector list
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import csv
from collections import namedtuple

Patient = namedtuple('Patient', ['age', 'gender', 'aspirin', 'steroids', 'noncebhaem', 'ischaemic', 'haemorrhagic', 'recurrent', 'haemodilution', 'thromb'])
# noncebheam is major non-cerebral haemoraging
# thromb is thrombolosis

# reads the data from the filename and returns the input data and the correct
# labels
def open_data(filename):
    # we first convert the patients into named tuples with their symptoms/etc.
    # then we turn them into vectors to be fed into the neural network

    patients = []
    patients_survived = []

    # generate the named tuples
    # the file is in Latin-1, not UTF-8
    with open(filename, 'r', encoding='latin-1') as patients_file:
        # we ignore the first line because it defines the headers, not a
        # patient
        patients_file.__next__() # intentionally ignored

        for patient in csv.reader(patients_file):
            patients.append(Patient(
                patient[4],
                patient[3],
                patient[10] or patient[27] or patient[85],
                patient[38],
                patient[43],
                patient[49],
                patient[50],
                # choose recurrent_{haem,isc} based on what they had
                patient[97] if patient[50] else patient[96],
                patient[40],
                patient[42]
                ))
            patients_survived.append(int(patient[65] == 'yes'))

    patient_vectors = []

    for patient in patients:
        patient_vector = (
                int(patient.age),
                int(patient.gender == 'Male'),
                int(patient.aspirin in ['yes', 'true']),
                int(patient.steroids == 'yes'),
                int(patient.noncebhaem == 'yes'),
                int(patient.ischaemic == 'yes'),
                int(patient.haemorrhagic == 'yes'),
                int(patient.recurrent == 'true'),
                int(patient.haemodilution == 'yes'),
                int(patient.thromb == 'yes'),
                )
        patient_vectors.append(patient_vector)

    return patient_vectors, patients_survived
