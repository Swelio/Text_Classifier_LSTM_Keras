#!/usr/bin/python3
# coding: utf-8

# ---- Description ----
""" Main script of project """

# ---- Imports ----
from classifier_class import Classifier

# ---- Script ----
if __name__ == '__main__':
    resources = '.'
    test = '.'

    detector = Classifier(resources,
                          weights_file='CNN_authors',
                          classifier_save_path='CNN_authors_detector',
                          fit_epochs=20,
                          data_size_max=3,
                          sequence_length=150)
    detector.display_prediction(test)
