#!/usr/bin/python3
# coding: utf-8

# ---- Description ----
""" A tool to classify texts with LSTM network """

# ---- Imports ----
import os
import sys
import glob
import numpy as np
import pickle
import random
from hashlib import sha3_224

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Embedding
from keras.preprocessing.text import text_to_word_sequence, hashing_trick
from keras.preprocessing.sequence import pad_sequences


# ---- Class ----
class Classifier:
    """ A tool to classify texts """

    def __init__(self,
                 resources_path,
                 classifier_save_path='my_classifier',
                 file_by_class=1,
                 data_size_max=50,
                 data_dir='Datas',
                 sequence_length=100,
                 total_vocab=None,
                 weights_file='weights.h5',
                 checkpoint=None,
                 fit_epochs=2,
                 overwrite=True,
                 reuse_datas=False):
        # --- INIT CONTEXT ---
        random.seed()
        if os.path.exists(classifier_save_path) and os.path.isfile(classifier_save_path):
            print('Classifier file already exists, loading it...')
            try:
                self.__dict__ = Classifier._load_classifier(classifier_save_path).__dict__.copy()
                self.load_weights(self.weights_file)
                print('Classifier successfully loaded')
                self.display_categories()
                return
            except EOFError as e:
                print('Failed to load classifier: File corrupted (EOFError)\n')

        # --- FILES ---
        self.classifier_save_path = classifier_save_path
        self.data_dir = data_dir
        self.byte_to_mb = 1048576
        self.data_size_max = self.byte_to_mb * data_size_max  # in bytes
        self.file_by_class = file_by_class  # files generated for each class

        # --- DATA TREATMENT ---
        self.sequence_length = sequence_length  # characters number
        self.total_vocab = total_vocab  # known letters number
        self.categories = []
        # initiate categories
        cat_dico = self._prepareCategories(resources_path=resources_path)
        # generate data files from resources for fitting
        self._generateDatas(texts_dico=cat_dico, overwrite=overwrite, reuse_datas=reuse_datas)

        # --- NEURAL NETWORK ---
        if not weights_file.endswith('.h5'):
            weights_file = weights_file.replace('.', '_')  # format secure file name
            weights_file += '.h5'
        self.weights_file = weights_file
        self.checkpoint = checkpoint
        self.optimizer = 'Nadam'
        self.loss = 'categorical_crossentropy'
        self.model = self._buildnet()  # build neural network model
        self.fit_on_all(epochs=fit_epochs)  # fit neural network from data files

        # --- STATE SAVE ---
        self.save_classifier()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = state['model'].to_json()
        state['weights'] = self.model.get_weights()
        state['checkpoint'] = None

        return state

    def __setstate__(self, state):
        state['model'] = model_from_json(state['model'])
        state['model'].set_weights(state['weights'])
        del state['weights']
        self.__dict__.update(state)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def save_classifier(self):
        with open(self.classifier_save_path, 'wb') as file:
            pickle.Pickler(file).dump(self)

    @staticmethod
    def _load_classifier(filename):
        with open(filename, 'rb') as file:
            classifier = pickle.Unpickler(file).load()
        return classifier

    def save_datas(self, datas, target, filename, overwrite=False):
        # --- PATH CONTROL ---
        if self.data_dir not in filename:
            filename = os.path.join(self.data_dir, filename)
        # --- OVERWRITING CONTROL ---
        if not overwrite:
            path_list = glob.glob(filename + '*')
            num = 0
            for path in path_list:
                temp = int(path[len(filename):])
                if temp > num:
                    num = temp + 1
            filename += str(num)
        # --- SAVING ---
        with open(filename, 'wb') as file:
            pickle.Pickler(file).dump([datas, target])

    def load_datas(self, filename=None):
        # --- PATH CONTROL ---
        if filename is None:
            dirList = os.listdir(self.data_dir)
            filename = random.choice(dirList)
        elif self.data_dir not in filename:
            filename = os.path.join(self.data_dir, filename)
        # --- READING DATAS ---
        with open(filename, 'rb') as file:
            datas, target = pickle.Unpickler(file).load()
        # --- RETURN ---
        return datas, target

    def _generateDatas(self, texts_dico, overwrite=True, reuse_datas=False):
        print('Generate files of {} MB each'.format(self.data_size_max / self.byte_to_mb))

        for category, text in texts_dico.items():
            targetIndex = self.categories.index(category)
            for files in range(self.file_by_class):
                print('[{}] File {} on {}'.format(category, files + 1, self.file_by_class))
                filename = os.path.join(self.data_dir, category + str(files))
                if reuse_datas:
                    if os.path.exists(filename):
                        continue
                datas, target = [], []
                while sys.getsizeof(np.array(datas)) < self.data_size_max:
                    piece = self.extract_datas(text)
                    datas.append(piece)

                    temp_target = [0.] * len(self.categories)
                    temp_target[targetIndex] = 1.
                    target.append(temp_target)
                self.save_datas(np.array(datas), np.array(target), filename, overwrite=overwrite)

        print('Datas extracted\n')

    @staticmethod
    def format_text(source_text):
        """
        Format text
        :param source_text: string
        :return: string
        """
        for sign in ('\n', '\ufeff', '\r'):
            source_text = source_text.replace(sign, ' ')
        return source_text

    @staticmethod
    def hash_function(x):
        return int(sha3_224(x.encode()).hexdigest(), 16)

    def transform_data(self, sequence):
        """ Transform a text into input data """
        data = hashing_trick(sequence, np.round(self.total_vocab * 1.3), hash_function=self.hash_function)
        data = pad_sequences([data], maxlen=self.sequence_length, padding='post')
        return np.reshape(data, data.shape[1:])

    def extract_datas(self, source_text, index=None):
        """ Used to extract a random sequence from a string """
        source_text = self.format_text(source_text=source_text)  # delete some useless characters
        try:
            temp_text = text_to_word_sequence(source_text)  # list of words
            if index is None:  # extract a random sequence from text
                index = random.randint(0, len(temp_text) - self.sequence_length)  # pick sequence index randomly
            data = temp_text[index:index + self.sequence_length]  # pick sequence
            temp_text = ""
            for word in data:  # sequence to string
                temp_text += word + ' '
            data = temp_text
        except ValueError:
            data = source_text

        return self.transform_data(data)

    def mix_datas(self):
        """ Used to extract datas randomly from data files (same amount per category) """
        datas, target = [], []

        for category in self.categories:
            temp_path = glob.glob(os.path.join(self.data_dir, category + '*'))
            temp_path = random.choice(temp_path)
            temp_datas, temp_target = self.load_datas(temp_path)
            number = temp_datas.shape[0] // len(self.categories)
            for i in range(number):
                index = random.randint(0, temp_datas.shape[0] - 1)
                datas.append(temp_datas[index])
                target.append(temp_target[index])

        datas, target = np.array(datas), np.array(target)

        return datas, target

    def _prepareCategories(self, resources_path):
        """ Register vocabulary and categories """
        texts_dico = {}
        superText = ''
        resources_dir = os.listdir(resources_path)

        # --- REGISTER CATEGORIES FROM SOURCES ---
        for dir_path in resources_dir:  # explore resources directory
            temp_path = os.path.join(resources_path, dir_path)
            if os.path.isdir(temp_path):  # look for a category directory
                texts_pathes = glob.glob(os.path.join(temp_path, '*.txt'))
                if len(texts_pathes) > 0:  # avoid empty folder
                    self.categories.append(dir_path)  # append category in knowns
                    category_text = ""
                    for text_file in texts_pathes:  # look for source texts
                        try:
                            # read text file
                            with open(text_file, 'r', encoding='utf-8') as file:
                                text = file.read()
                            text = self.format_text(source_text=text)
                            category_text += text  # append text to the global category text
                        except (FileNotFoundError, OSError) as e:
                            print(e)
                    superText += category_text + ' '  # append the global category text to the global text
                    texts_dico[dir_path] = category_text  # register category text into knowns categories

        self.display_categories()  # display knowns categories

        actual_vocab = len(set(text_to_word_sequence(superText)))
        if self.total_vocab is None:
            self.total_vocab = actual_vocab

        print("Total vocabulary:", actual_vocab)
        return texts_dico

    def display_categories(self):
        print('Total categories: {}'.format(len(self.categories)))
        for category in self.categories:
            end = ' - '
            if self.categories.index(category) == len(self.categories) - 1:
                end = '\n' * 2
            print(category, end=end)

    def _buildnet(self):
        """
        Build the neural network model and load weights from weight file if exists
        :return: neural network
        """

        model = Sequential()
        model.add(Embedding(self.total_vocab + 1,  # limitation memory
                            int(np.round(self.sequence_length * 1.5)),
                            input_length=int(self.sequence_length)))
        model.add(Conv1D(8, 3, activation='relu'))
        model.add(Dropout(0.01))
        model.add(MaxPooling1D())
        model.add(Conv1D(16, 3, activation='relu'))
        model.add(Dropout(0.01))
        model.add(MaxPooling1D())
        model.add(LSTM(64, recurrent_dropout=0.01))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.01))
        model.add(Dense(len(self.categories), activation='softmax'))
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        try:
            model.load_weights(self.weights_file)
        except (OSError, ValueError) as e:
            print(e, end='\n' * 2)

        return model

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def fit_on_all(self, epochs=2):
        """ Fit neural network from data files """
        if self.checkpoint is None:
            # create a checkpoint callback
            self.checkpoint = ModelCheckpoint(self.weights_file, monitor='loss',
                                              save_best_only=True,
                                              save_weights_only=True,
                                              mode='min',
                                              verbose=1)
        # --- FITTING LOOP ---
        for i in range(1, epochs + 1):
            print('\nTraining epoch: {} / {}'.format(i, epochs))
            datas, target = self.mix_datas()
            x_test, y_test = self.mix_datas()

            self.model.fit(datas, target,
                           batch_size=16, epochs=1,
                           validation_data=(x_test, y_test),
                           verbose=1,
                           callbacks=[self.checkpoint])
            self.save_classifier()  # save classifier
        self.load_weights(self.weights_file)
        self.save_weights(self.weights_file)
        print()

    def predict(self, filepath):
        """ Make a prediction from a text file """
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()  # get source text

        totalWords = len(text_to_word_sequence(text))

        results = np.full((1,) + self.model.output_shape[1:], 0.)
        for i in range(totalWords - self.sequence_length):
            data = self.extract_datas(source_text=text, index=i)  # extract some data randomly from text file
            data = data.reshape((1,) + data.shape)  # prepare it
            results += self.model.predict(data)  # make a prediction
        prediction = results / (
                totalWords - self.sequence_length)  # make the average (to avoid error with bad extracts)
        return prediction.reshape(prediction.shape[1:])

    def display(self, filepath, limit=0):
        """ Display prediction """
        # prepare a display name
        name = os.path.join(os.path.basename(os.path.dirname(filepath)), os.path.basename(filepath))
        prediction = self.predict(filepath)  # predict

        results = []
        for i in range(len(prediction)):  # sort predictions
            results.append([prediction[i], self.categories[i]])
        results = sorted(results, key=lambda res: res[0], reverse=True)

        if not limit:
            limit = len(results)

        print('Predictions for:', name)
        for e in results[:limit]:  # display each category with its result
            print("{0:.2f}% - {1}".format(np.round(e[0] * 100, 2), e[1]))
        print()

    def display_prediction(self, file, limit=0):
        """ Display predictions for a list of texts """
        if type(file) in (tuple, list):
            for path in file:
                self.display(path, limit=limit)
        elif type(file) is str:
            if os.path.isdir(file):
                files = glob.glob(os.path.join(file, '*.txt'))
                self.display_prediction(files, limit=limit)
            else:
                self.display(file, limit=limit)
