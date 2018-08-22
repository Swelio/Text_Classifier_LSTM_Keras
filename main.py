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
                          weights_file='CNN_languages',
                          classifier_save_path='CNN_languages_detector',
                          fit_epochs=30,
                          data_size_max=70)
    detector.display_prediction(test)
