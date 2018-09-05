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
    test2 = '.'

    detector = Classifier(resources,
                          weights_file='CNN_authors',
                          classifier_save_path='CNN_authors_detector',
                          data_per_categorie=1.,  # amount of data loaded in batch for each category
                          fit_epochs=40,
                          data_size_max=5,
                          sequence_length=200,
                          reuse_datas=False)
    detector.display_prediction([test, test2])
