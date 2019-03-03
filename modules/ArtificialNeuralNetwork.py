import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from modules.TextPreprocessor import TextPreProcessor
ep = 40
batch = 200


class DNN:
    def __init__(self, components=100, labels=3, load=False, path='.'):
        if not load:
            self.file_path = path
            self.model = keras.Sequential()
            self.model.add(keras.layers.Dense(16,input_dim =components, activation=tf.nn.relu))
            # self.model.add(keras.layers.Dense(16, activation=tf.nn.relu))
            self.model.add(keras.layers.Dense(16, activation=tf.nn.relu))
            # self.model.add(keras.layers.Dense(16, activation=tf.nn.relu))
            self.model.add(keras.layers.Dense(labels, activation=tf.nn.softmax))
            self.summary = self.model.summary()
        else:
            self.model = keras.models.load_model(self.file_path+'/EncapsulatedFiles/DNNClassifier.h5')
            self.summary = self.model.summary()

    def compile(self):
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    def fit(self,X_train, y_train, epochs=ep, batch_size=batch):
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)


    def predict(self, X_test):
        return self.model.predict(X_test)


    def save(self, filename = '/EncapsulatedFiles/DNNClassifier.h5'):
        self.model.save(self.file_path + filename)




#Embedding Neural Network
class ENN:
    tpp = TextPreProcessor()

    def __init__(self, train=[],labels=3, load=False, max_length=15, path='.'):
        if not load:
            self.file_path = path
            self.dict_hash = {}
            self.train = self.TextPreprocess_fit(train)
            self.model = keras.Sequential()
            self.model.add(keras.layers.Embedding(input_dim=self.vocab_size, output_dim=30, input_length =self.input_length))
            # self.model.add(keras.layers.LSTM(50))
            # self.model.add(keras.layers.Flatten())
            self.model.add(keras.layers.GlobalAveragePooling1D())
            self.model.add(keras.layers.Dense(32, activation=tf.nn.relu))
            # self.model.add(keras.layers.Dense(32, activation=tf.nn.relu))
            self.model.add(keras.layers.Dense(16, activation=tf.nn.relu))
            self.model.add(keras.layers.Dense(labels, activation=tf.nn.softmax))
            self.summary = self.model.summary()
        else:
            self.dict_hash = {}
            self.model = keras.models.load_model(self.file_path + '/EncapsulatedFiles/ENNClassifier.h5')
            self.summary = self.model.summary()
            self.input_length = self.model.get_config()[0]['config']['batch_input_shape'][1]
            self.vocab_size = self.model.get_config()[0]['config']['input_dim']

    def compile(self):
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X_train, y_train, epochs=ep, batch_size=batch):
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        return self.model.predict(X_test)


    def save(self, filename = '/EncapsulatedFiles/ENNClassifier.h5'):
        self.model.save(self.file_path + filename)




    # require to have pre-processed the input data
    def TextPreprocess_fit(self, train):
        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(train)
        self.vocab_size = len(tokenizer.word_counts) + 1

        # replace words with an identifier (counter) and store the produced dictionary
        hash_train = tokenizer.texts_to_sequences(train)
        self.dict_hash = tokenizer.word_index
        pickle.dump(self.dict_hash, open(self.file_path + '/EncapsulatedFiles/HashDictionary', 'wb'))

        # padding with 0 in order the lists will be in the same size
        self.input_length = int(max([len(t) for t in hash_train]))
        train = keras.preprocessing.sequence.pad_sequences(hash_train, maxlen=self.input_length, padding='post')

        return train



    # require to have pre-processed the input data
    def TextPreprocess_transform(self, test):
        if len(self.dict_hash) == 0 :
            self.dict_hash = pickle.load(open(self.file_path + '/EncapsulatedFiles/HashDictionary', 'rb'))

        # use  produced dictionary to convert text into list of ints and also
        # count the number of unknown words -- The dict is constructed in fit()
        hash_test = []
        unknown_words = 0
        for text in test:
            hash_text = []
            for word in text.split(' '):
                if word in self.dict_hash:
                    hash_text.append(self.dict_hash[word])
                else:
                    unknown_words += 1
            hash_test.append(hash_text)
        print("Unknown Words :\t", unknown_words)

        test = keras.preprocessing.sequence.pad_sequences(hash_test, maxlen=self.input_length, padding='post')
        return test
