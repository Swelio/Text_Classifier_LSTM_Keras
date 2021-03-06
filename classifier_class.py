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
import string

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
                 data_per_batch=10.,
                 data_size_max=50,
                 data_dir='Datas',
                 sequence_length=100,
                 total_vocab=None,
                 weights_file='weights.h5',
                 checkpoint=None,
                 fit_epochs=2,
                 overwrite=True,
                 reuse_datas=False,
                 letter_mode=False):
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

        # --- DATA TREATMENT ---
        self.letter_mode = letter_mode
        self.data_per_batch = self.byte_to_mb * data_per_batch  # amount of data per categorie in batch
        self.sequence_length = sequence_length  # characters number
        self.total_vocab = total_vocab  # known letters number
        self.categories = {}
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
        self.optimizer = 'rmsprop'
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
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

    def save_classifier(self):
        with open(self.classifier_save_path, 'wb') as file:
            pickle.Pickler(file).dump(self)

    @staticmethod
    def _load_classifier(filename):
        with open(filename, 'rb') as file:
            classifier = pickle.Unpickler(file).load()
        return classifier

    def save_datas(self, datas, filename, overwrite=False):
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
            pickle.Pickler(file).dump(datas)

    def load_datas(self, filename=None):
        # --- PATH CONTROL ---
        if filename is None:
            dirList = os.listdir(self.data_dir)
            filename = random.choice(dirList)
        elif self.data_dir not in filename:
            filename = os.path.join(self.data_dir, filename)
        # --- READING DATAS ---
        with open(filename, 'rb') as file:
            datas = pickle.Unpickler(file).load()
        # --- RETURN ---
        return datas

    def _generateDatas(self, texts_dico, overwrite=True, reuse_datas=False):
        print('Generate files of {} MB maximum for each'.format(self.data_size_max / self.byte_to_mb))

        for category, textList in texts_dico.items():
            print('[ {0} ] '.format(category), end='', flush=True)
            index = 0  # index of text from pick data
            count = [0] * len(textList)  # indexes for each text of sequences
            completed = 0  # number of files extracted
            created_files = 0  # number of data files created
            while completed < len(textList):  # while each text has not been totally read
                filename = os.path.join(self.data_dir, category + str(created_files))
                if reuse_datas:
                    if os.path.exists(filename):
                        created_files += 1
                        break
                datas = []
                while completed < len(textList) and sys.getsizeof(np.array(datas)) < self.data_size_max:
                    text = textList[index]  # pick sequence of each text
                    if count[index] <= len(text) - self.sequence_length:
                        piece = self.extract_datas(text, count[index])
                        count[index] += self.sequence_length
                        datas.append(piece)
                    else:
                        completed += 1
                    index = (index + 1) % len(textList)  # go to next text
                self.save_datas(np.array(datas), filename, overwrite=overwrite)
                created_files += 1
            print('-> {0} File{1} created'.format(created_files, chr(ord('s') * (created_files - 1))))
        print('Data extracted\n')

    @staticmethod
    def format_text(source_text):
        """
        Format text
        :param source_text: string
        :return: string
        """
        for sign in ('\n', '\ufeff', '\r'):
            source_text = source_text.replace(sign, ' ')
        return source_text.lower()

    @staticmethod
    def hash_function(x):
        return int(sha3_224(x.encode()).hexdigest(), 16)

    def transform_data(self, sequence):
        """ Transform a text into input data """
        data = hashing_trick(sequence, np.round(self.total_vocab * 1.3), hash_function=self.hash_function)
        data = pad_sequences([data], maxlen=self.sequence_length, padding='post')
        return np.reshape(data, data.shape[1:])

    def extract_datas(self, source_text, index=0):
        """ Used to extract a sequence from a string """
        source_text = self.format_text(source_text=source_text)  # delete some useless characters
        if self.letter_mode:
            temp_text = source_text
            data = ""
            i = index
            while i < len(temp_text) and len(data) < self.sequence_length:
                if temp_text[i] != ' ':  # skip spaces
                    if temp_text[i] in string.punctuation:  # add punctuation individually
                        data += temp_text[i]
                    else:
                        # --- MEASURE WORD ---
                        long = 0
                        while i + long < len(temp_text) and temp_text[i + long] not in string.punctuation + ' ':
                            long += 1
                        if i + long < len(temp_text) and len(data) + long <= self.sequence_length:
                            data += temp_text[i:i + long]
                            i += long - 1
                        elif len(data) + long > self.sequence_length:
                            break
                i += 1
        else:
            temp_text = text_to_word_sequence(source_text)  # list of words
            data = temp_text[index:index + self.sequence_length]  # pick sequence
            temp_text = ""
            for word in data:  # sequence to string
                temp_text += word + ' '
            data = temp_text

        return self.transform_data(data)

    def mix_datas(self):
        """ Used to extract datas from data files (same amount per category if possible) """

        datas, target = [], []
        max_per_category = self.data_per_batch / len(self.categories)

        for category in self.categories.keys():  # for each known category
            total_files = len(glob.glob(os.path.join(self.data_dir, category + '*')))  # count all files of category
            temp_datas = self._load_from_file(category)  # load datas from first file
            cat_datas = []
            # --- DATA EXTRACTION ---
            while (sys.getsizeof(np.array(cat_datas)) < max_per_category  # byte size limit for memory secure
                   and self.categories.get(category).get('fileIndex') < total_files):  # each file read one time

                index = self.categories.get(category).get('inFileIndex')  # index of sequence in data
                self.categories.get(category)['inFileIndex'] += 1  # prepare the next index

                if index == len(temp_datas):  # all data in file has been read
                    self.categories.get(category)['fileIndex'] += 1  # go to next data file
                    self.categories.get(category)['inFileIndex'] = 0  # initialize reading at first index
                    try:
                        temp_datas = self._load_from_file(category)  # load datas from file
                    except FileNotFoundError:
                        continue
                else:
                    # verify if the raw is already in the batch
                    c = True
                    for raw in cat_datas:
                        if np.array_equal(raw, temp_datas[index]):
                            c = False
                            break
                    if c:  # if not already in data
                        cat_datas.append(temp_datas[index])  # extract one sequence
            if self.categories.get(category).get('fileIndex') == total_files:  # go at the beginning of the first file
                self.categories.get(category)['fileIndex'] = 0
                self.categories.get(category)['inFileIndex'] = 0

            # --- PREPARING TARGET ---
            temp_target = [0.] * len(self.categories)
            temp_target[self.categories.get(category).get('index')] = 1.
            temp_target = [temp_target] * len(cat_datas)

            # --- ADD TARGET TO LOCAL BATCH ---
            target += temp_target

            # --- ADD DATA TO GLOBAL BATCH ---
            datas += cat_datas

        # --- CONVERT TO NUMPY ARRAYS ---
        datas, target = np.array(datas), np.array(target)

        return datas, target

    def _load_from_file(self, category):
        temp_path = os.path.join(self.data_dir, category + str(self.categories.get(category).get('fileIndex')))
        return self.load_datas(temp_path)

    def _prepareCategories(self, resources_path):
        """ Register vocabulary and categories """
        texts_dico = {}
        superText = ''
        resources_dir = os.listdir(resources_path)
        cat_index = 0

        # --- REGISTER CATEGORIES FROM SOURCES ---
        for dir_path in resources_dir:  # explore resources directory
            temp_path = os.path.join(resources_path, dir_path)
            if os.path.isdir(temp_path):  # look for a category directory
                texts_pathes = glob.glob(os.path.join(temp_path, '*.txt'))
                if len(texts_pathes) > 0:  # avoid empty folder
                    self.categories[dir_path] = {'index': cat_index,  # append category in knowns
                                                 'fileIndex': 0,
                                                 'inFileIndex': 0}
                    cat_index += 1
                    category_text = []
                    for text_file in texts_pathes:  # look for source texts
                        try:
                            # read text file
                            with open(text_file, 'r', encoding='utf-8') as file:
                                text = file.read()
                            text = self.format_text(source_text=text)
                            category_text.append(text)
                            superText += text + ' '
                        except (FileNotFoundError, OSError) as e:
                            print(e)
                    texts_dico[dir_path] = category_text  # register category text into knowns categories

        self.display_categories()  # display knowns categories

        if self.letter_mode:
            actual_vocab = len(set(superText.replace(' ', '')))
        else:
            actual_vocab = len(set(text_to_word_sequence(superText)))
        if self.total_vocab is None:
            self.total_vocab = actual_vocab

        print("Total vocabulary: {} (maximum: {})".format(actual_vocab, self.total_vocab))
        return texts_dico

    def display_categories(self):
        print('Total categories: {}'.format(len(self.categories)))
        for category in self.categories.keys():
            end = ' - '
            if self.categories.get(category).get('index') == len(self.categories) - 1:
                end = '\n' * 2
            print(category, end=end)

    def _buildnet(self):
        """
        Build the neural network model and load weights from weight file if exists
        :return: neural network
        """

        model = Sequential()
        model.add(Embedding(self.total_vocab + 1,
                            int(np.round(self.sequence_length * 1.5)),
                            input_length=int(self.sequence_length)))
        model.add(Conv1D(4, 3, activation='relu'))
        model.add(Dropout(0.05))
        model.add(MaxPooling1D())
        model.add(Conv1D(8, 3, activation='relu'))
        model.add(Dropout(0.05))
        model.add(MaxPooling1D())
        model.add(Conv1D(8, 3, activation='relu'))
        model.add(Dropout(0.05))
        model.add(MaxPooling1D())
        model.add(Conv1D(16, 3, activation='relu'))
        model.add(Dropout(0.05))
        model.add(MaxPooling1D())
        model.add(LSTM(64, return_sequences=False, recurrent_dropout=0.05))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.05))
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
        oneBatch = False
        for i in range(1, epochs + 1):
            print('\nTraining epoch: {} / {}'.format(i, epochs))
            next_datas, next_target = [], []
            if not oneBatch:
                next_datas, next_target = self.mix_datas()
            while True:  # loop on all data
                datas, target = next_datas, next_target
                if len(datas) > 0:
                    self.model.fit(datas, target,
                                   batch_size=16, epochs=1,
                                   verbose=1,
                                   callbacks=[self.checkpoint])
                self.save_classifier()  # save classifier

                if not oneBatch:
                    next_datas, next_target = self.mix_datas()
                    if np.array_equal(datas, next_datas):
                        oneBatch = True

                if len(next_datas) == 0 or oneBatch:
                    for value in self.categories.values():
                        value['fileIndex'] = 0
                        value['inFileIndex'] = 0
                    break
        self.load_weights(self.weights_file)
        self.save_weights(self.weights_file)
        print()

    @staticmethod
    def search_index(dico, index):
        for key, values in dico.items():
            if index == values.get('index'):
                return key
        return ''

    def predict(self, filepath):
        """ Make a prediction from a text file """
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()  # get source text
        text = self.format_text(text)

        totalWords = len(text_to_word_sequence(text))

        prediction = np.full((1,) + self.model.output_shape[1:], 0.)
        count = 0
        for i in range(0, totalWords, self.sequence_length):
            data = self.extract_datas(source_text=text, index=i)  # extract some data randomly from text file
            data = data.reshape((1,) + data.shape)  # prepare it
            prediction += self.model.predict(data)  # make a prediction
            count += 1
        if count > 0:
            prediction /= count  # make the average (to avoid error with bad extracts)
        else:
            print('No extraction')
        return prediction.reshape(prediction.shape[1:])

    def display(self, filepath, limit=0):
        """ Display prediction """
        # prepare a display name
        name = os.path.join(os.path.basename(os.path.dirname(filepath)), os.path.basename(filepath))
        prediction = self.predict(filepath)  # predict

        results = []
        for i in range(len(prediction)):  # sort predictions
            results.append([prediction[i], self.search_index(self.categories, i)])
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
                self.display_prediction(path, limit=limit)
        elif type(file) is str:
            if os.path.isdir(file):
                files = os.listdir(file)
                for i in range(len(files)):
                    files[i] = os.path.join(file, files[i])
                self.display_prediction(files, limit=limit)
            elif file.endswith('.txt'):
                self.display(file, limit=limit)
