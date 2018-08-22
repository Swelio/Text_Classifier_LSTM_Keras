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

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer


# ---- Class ----
class Classifier:
    """ A tool to classify texts """

    def __init__(self,
                 resources_path,
                 classifier_save_path='my_classifier',
                 file_by_class=1,
                 data_size_max=50,
                 data_dir='Datas',
                 sequence_length=75,
                 total_vocab=None,
                 weights_file='weights.h5',
                 checkpoint=None,
                 fit_epochs=2,
                 overwrite=True):
        # --- INIT CONTEXT ---
        random.seed()
        if os.path.exists(classifier_save_path) and os.path.isfile(classifier_save_path):
            print('Classifier file already exists, loading it...')
            try:
                self.__dict__ = Classifier._load_classifier(classifier_save_path).__dict__.copy()
                print('Classifier successfully loaded')
                self.display_categories()
                return
            except EOFError as e:
                print('Failed to load classifier:', e, '\n')

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
        self.tokenizer, cat_dico = self._learnLetters(resources_path=resources_path)
        self._generateDatas(texts_dico=cat_dico, overwrite=overwrite)

        # --- NEURAL NETWORK ---
        if not weights_file.endswith('.h5'):
            weights_file = weights_file.replace('.', '_')
            weights_file += '.h5'
        self.weights_file = weights_file
        self.checkpoint = checkpoint
        self.model = self._buildnet()
        self.fit_on_all(epochs=fit_epochs)

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

    def save_classifier(self):
        self.save_weights(self.weights_file)
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

    def _generateDatas(self, texts_dico, overwrite=True):
        print('Generate files of {} MB each'.format(self.data_size_max / self.byte_to_mb))

        for category, text in texts_dico.items():
            targetIndex = self.categories.index(category)
            for files in range(self.file_by_class):
                print('[{}] File {} on {}'.format(category, files + 1, self.file_by_class))
                datas, target = [], []
                while sys.getsizeof(np.array(datas)) < self.data_size_max:
                    i = random.randint(0, len(text) - self.sequence_length)

                    piece = text[i:i + self.sequence_length]
                    piece = self.tokenizer.texts_to_matrix(piece)
                    datas.append(piece)

                    temp_target = [0.] * len(self.categories)
                    temp_target[targetIndex] = 1.
                    target.append(temp_target)
                self.save_datas(np.array(datas), np.array(target), category + str(files), overwrite=overwrite)

        print('Datas extracted\n')

    def extract_datas(self, source_text):
        for sign in ('\n', '\ufeff', '\r', ' '):
            source_text = source_text.replace(sign, '')
        try:
            i = random.randint(0, len(source_text) - self.sequence_length)
            data = source_text[i:i + self.sequence_length]
        except ValueError:
            data = source_text
        data = self.tokenizer.texts_to_matrix(data)
        return data

    def mix_datas(self):
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

    def _learnLetters(self, resources_path):
        tokenizer = Tokenizer(num_words=self.total_vocab)
        texts_dico = {}
        superText = ""

        resources_dir = os.listdir(resources_path)

        for dir_path in resources_dir:
            temp_path = os.path.join(resources_path, dir_path)
            if os.path.isdir(temp_path):
                self.categories.append(dir_path)
                texts_pathes = glob.glob(os.path.join(temp_path, '*.txt'))
                category_text = ""
                for text_file in texts_pathes:
                    try:
                        with open(text_file, 'r', encoding='utf-8') as file:
                            text = file.read()
                        for sign in ('\n', '\ufeff', '\r', ' '):
                            text = text.replace(sign, '')
                        category_text += text
                    except (FileNotFoundError, OSError) as e:
                        print(e)
                superText += category_text
                texts_dico[dir_path] = category_text

        self.display_categories()

        tokenizer.fit_on_texts(superText)

        return tokenizer, texts_dico

    def display_categories(self):
        print('Total categories: {}'.format(len(self.categories)))
        for category in self.categories:
            end = ' - '
            if self.categories.index(category) == len(self.categories) - 1:
                end = '\n' * 2
            print(category, end=end)

    def _buildnet(self):
        if self.total_vocab is None:
            self.total_vocab = len(self.tokenizer.word_index) + 1
        model = Sequential()
        model.add(Conv1D(8, 3, activation='relu', input_shape=(self.sequence_length, self.total_vocab)))
        model.add(MaxPooling1D())
        model.add(Conv1D(16, 3, activation='relu'))
        model.add(MaxPooling1D())
        model.add(Conv1D(32, 3, activation='relu'))
        model.add(MaxPooling1D())
        # model.add(LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.total_vocab),
        #                recurrent_dropout=0.05))
        model.add(LSTM(64, return_sequences=False, recurrent_dropout=0.05))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(len(self.categories), activation='softmax'))
        model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

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
        if self.checkpoint is None:
            self.checkpoint = ModelCheckpoint(self.weights_file, monitor='loss',
                                              save_best_only=True,
                                              save_weights_only=True,
                                              mode='min',
                                              verbose=1)
        for i in range(1, epochs + 1):
            print('\nTraining epoch: {} / {}'.format(i, epochs))
            datas, target = self.mix_datas()
            x_test, y_test = self.mix_datas()

            self.model.fit(datas, target,
                           batch_size=16, epochs=10,
                           validation_data=(x_test, y_test),
                           verbose=1,
                           callbacks=[self.checkpoint])
        print()

    def predict(self, filepath, num=30):
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()

        data = self.extract_datas(source_text=text)
        data = data.reshape((1,) + data.shape)
        results = self.model.predict(data)
        num = num
        for i in range(num - 1):
            data = self.extract_datas(source_text=text)
            data = data.reshape((1,) + data.shape)
            results += self.model.predict(data)
        prediction = results / num
        return int(np.argmax(prediction[0]))

    def display_prediction(self, filepath, num=30):
        print("[{}] {}".format(self.categories[self.predict(filepath, num=num)],
                               os.path.join(os.path.basename(os.path.dirname(filepath)), os.path.basename(filepath))))
