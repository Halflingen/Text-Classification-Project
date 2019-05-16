#A CNN made with Keras
from keras.layers import Input, Dense, Flatten, Dropout, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils.vis_utils import plot_model
from keras.models import Model, load_model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.utils.np_utils import to_categorical
import time
import pickle
from keras import backend as K

import get_data
import metrics


class CNN:
    def __init__(self):
        self.data = get_data.Data()
        self.train_data,\
        self.train_target,\
        self.test_data,\
        self.test_target\
        = self.data.fetch_train_data()

        self.tokenizer = self.create_tokenizer()
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.para_len = max([len(s.split()) for s in self.train_data])

        self.encoded_train = self.tokenizer.texts_to_sequences(self.train_data)
        self.padded_train = pad_sequences(self.encoded_train, maxlen=self.para_len, padding='post')


    def create_tokenizer(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.train_data)
        return tokenizer


    def create_model(self, para):
        '''
        Parameters:
        0 - emb_dim
        1 - filter_size
        2 - kernel_size
        3 - pool_size
        4 - dropout_rate
        5 - dense_layer
        '''
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=para[0], input_length=self.para_len))
        model.add(Conv1D(filters=para[1], kernel_size=para[2], activation='relu'))
        model.add(Dropout(rate=para[4]))
        model.add(MaxPooling1D(pool_size=para[3]))
        model.add(Flatten())
        model.add(Dense(units=para[5], activation='relu'))
        model.add(Dense(units=5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


    def train_model(self, parameters):
        '''
        Parameters:
        0 - emb_dim
        1 - filter_size
        2 - kernel_size
        3 - pool_size
        4 - dropout_rate
        5 - dense_layer
        '''

        all_para = (*parameters, 'adam', 10, 32)
        metric = metrics.MetricsAndParameters('cnn', all_para)

        if(metric.duplicate()):
            return


        skf = StratifiedKFold(n_splits=10, shuffle=False)
        for train_index, test_index in skf.split(self.padded_train, self.train_target):
            #Data
            train_data = self.padded_train[train_index]
            train_target = to_categorical(self.train_target[train_index],num_classes=5)
            test_data = self.padded_train[test_index]
            test_target = self.train_target[test_index]

            #Training
            train_time = time.time()
            model = self.create_model(parameters)
            model.fit([train_data], train_target, epochs=10, batch_size=32)
            train_time =  time.time() - train_time

            #Prediction
            prediction_time = time.time()
            score = model.predict(test_data, batch_size=32)
            prediction_time =  time.time() - prediction_time

            #Metric calculation
            test_results = score.argmax(axis=-1)
            metric.calculate_metrics(test_target, test_results, train_time, prediction_time)

        #save to database
        metric.store_metrics()
        metric.store_parameters()
        K.clear_session()


    def find_optimal_parameters(self, parameters):
        for a in parameters[0]:
            for b in parameters[1]:
                for c in parameters[2]:
                    for d in parameters[3]:
                        for e in parameters[4]:
                            for f in parameters[5]:
                                    self.train_model((a,b,c,d,e,f))


    def create_and_save_model(self, parameters):
        train_data = self.padded_train
        train_target = to_categorical(self.train_target,num_classes=5)

        model = self.create_model(parameters)
        model.fit([train_data], train_target, epochs=10, batch_size=32)

        model.save('model_cnn.h5')
        with open('tokenizer_cnn.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    cnn = CNN()

    #Finde optimal embedding dimension
    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [32, 64, 128, 256]
    kernel_size = [3,6]
    pool_size = [2]
    dropout_rate = [0.5]
    dense_layer = [25]
    parameters = \
            (embedding_dimension,
            filter_size,
            kernel_size,
            pool_size,
            dropout_rate,
            dense_layer)

    cnn.find_optimal_parameters(parameters)

    embedding_dimension = [i for i in range(10, 250, 3)]
    filter_size = [32, 64, 128, 256]
    kernel_size = [2,3,4,8]
    pool_size = [4]
    dropout_rate = [0.5]
    dense_layer = [25]
    parameters = \
            (embedding_dimension,
            filter_size,
            kernel_size,
            pool_size,
            dropout_rate,
            dense_layer)

    cnn.find_optimal_parameters(parameters)

    #Finde optimal filter size
    embedding_dimension = [210]
    filter_size = [i for i in range(10,256,3)]
    kernel_size = [6]
    pool_size = [2]
    dropout_rate = [0.5]
    dense_layer = [25]
    parameters = \
            (embedding_dimension,
            filter_size,
            kernel_size,
            pool_size,
            dropout_rate,
            dense_layer)

    cnn.find_optimal_parameters(parameters)

    #Finde optimal Dense layer
    embedding_dimension = [210]
    filter_size = [182]
    kernel_size = [6]
    pool_size = [2]
    dropout_rate = [0.5]
    dense_layer = [i for i in range(10, 250, 3)]
    parameters = \
            (embedding_dimension,
            filter_size,
            kernel_size,
            pool_size,
            dropout_rate,
            dense_layer)

    cnn.find_optimal_parameters(parameters)

    #Finde optimal dropout rate
    embedding_dimension = [210]
    filter_size = [182]
    kernel_size = [6]
    pool_size = [2]
    dropout_rate = [i for i in np.arange(0.01, 1.0, 0.0125)]
    dense_layer = [178]
    parameters = \
            (embedding_dimension,
            filter_size,
            kernel_size,
            pool_size,
            dropout_rate,
            dense_layer)

    cnn.find_optimal_parameters(parameters)


    #Creating the optimal model
    #para = (210, 182, 6, 2, 0.09, 'adam', 178, 10, 32)
    #cnn.create_and_save_model(para)


if __name__ == '__main__':
    main()
