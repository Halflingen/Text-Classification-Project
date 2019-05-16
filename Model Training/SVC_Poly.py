#SVC doesent work on the dataset. There are to many samples. LinearSVC works
#and perfomrs slighly better than naive bayes
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
import time
import numpy as np
import pickle

import get_data
import metrics


class SVC_:
    def __init__(self):
        self.data = get_data.Data()
        self.train_data,\
        self.train_target,\
        self.test_data,\
        self.test_target\
        = self.data.fetch_train_data()


    def create_model(self, para):
        '''
        0 - c
        1 - gamma
        2 - shrinking
        3 - coef0
        '''
        vectorizer = TfidfVectorizer()

        clf = SVC(C=para[0], gamma=para[1], kernel='poly', coef0=para[3],
                degree=2, shrinking=para[2])
        pipeline = Pipeline([
            ('vec', vectorizer),
            ('clf', clf),
        ])

        return pipeline


    def train_model(self, parameters):
        '''
        0 - c
        1 - gamma
        2 - shrinking
        3 - coef0
        '''

        all_para = ('tfidf', *parameters, 1e-3, 'poly', 2)
        metric = metrics.MetricsAndParameters('svc', all_para)

        if(metric.duplicate()):
            return

        skf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, test_index in skf.split(self.train_data, self.train_target):
            #Data
            train_data = self.train_data[train_index]
            train_target = self.train_target[train_index]
            test_data = self.train_data[test_index]
            test_target = self.train_target[test_index]

            #Training
            train_time = time.time()
            model = self.create_model(parameters)
            model.fit(train_data, train_target)
            train_time =  time.time() - train_time

            #Prediction
            prediction_time = time.time()
            test_results = model.predict(test_data)
            prediction_time =  time.time() - prediction_time

            #Metric calculation
            metric.calculate_metrics(test_target, test_results, train_time, prediction_time)

        #save to database
        metric.store_metrics()
        metric.store_parameters()


    def find_optimal_parameters(self, parameters):
        for a in parameters[0]:
            for b in parameters[1]:
                for c in parameters[2]:
                    for d in parameters[3]:
                        self.train_model((a,b,c,d))


    def create_and_save_model(self, parameters):
        train_data = self.train_data
        train_target = self.train_target

        model = self.create_model(parameters)
        model.fit(train_data, train_target)

        with open("model_svc_poly.pickle", 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    svc = SVC_()

    #Finding optimal c value
    c = [i for i in np.arange(0.5,1.5,0.0125)]
    gamma = [1]
    shrinking = [True, False]
    coef0 = [0.0]
    parameters = \
            (c,
            gamma,
            shrinking,
            coef0)

    svc.find_optimal_parameters(parameters)

    #Finding optimal gamma value
    c = [0.96]
    gamma = [i for i in np.arange(0.5,1.5,0.0125)]
    shrinking = [True, False]
    coef0 = [0.0]
    parameters = \
            (c,
            gamma,
            shrinking,
            coef0)

    svc.find_optimal_parameters(parameters)

    #Finding optimal Coef0 value
    c = [0.96]
    gamma = [1.2]
    shrinking = [True, False]
    coef0 = [i for i in np.arange(0.00,3,0.0375)]
    parameters = \
            (c,
            gamma,
            shrinking,
            coef0)

    svc.find_optimal_parameters(parameters)

    #para = (0.96, 1.2, False, 0.63)
    #svc.create_and_save_model(para)


if __name__ == '__main__':
    main()
