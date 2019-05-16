from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
import time
import numpy as np
import pickle

import get_data
import metrics


class LinearSVC_:
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
        1 - tol
        2 - intercept_scaling
        '''
        vectorizer = TfidfVectorizer()

        clf = LinearSVC(C=para[0], tol=para[1], intercept_scaling=para[2], max_iter=5000)
        pipeline = Pipeline([
            ('vec', vectorizer),
            ('clf', clf),
        ])

        return pipeline

    def train_model(self, parameters):
        '''
        0 - c
        1 - tol
        2 - intercept_scaling
        '''

        all_para = ('tfidf', *parameters, 'l2', 'squared_hinge', True, 'ovr', True)
        metric = metrics.MetricsAndParameters('linear_svc', all_para)

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
                    self.train_model((a,b,c))


    def create_and_save_model(self, parameters):
        train_data = self.train_data
        train_target = self.train_target

        model = self.create_model(parameters)
        model.fit(train_data, train_target)

        with open('t2_model_linearSVC.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    linear_svc = LinearSVC_()

    #Find Optimal C value
    c1 = [i for i in np.arange(0.005, 0.25, 0.008)]
    c2 = [i for i in np.arange(0.25, 1.5, 0.02)]
    c = c1+c2
    tol = [1e-4]
    intercept_scaling = [2]
    parameters = \
            (c,
            tol,
            intercept_scaling)

    linear_svc.find_optimal_parameters(parameters)

    #Find Optimal intercept scaling value
    c = [0.052]
    tol = [1e-4]
    intercept_scaling = [i for i in np.arange(1, 10, 0.2)]
    parameters = \
            (c,
            tol,
            intercept_scaling)

    linear_svc.find_optimal_parameters(parameters)

    #Find Optimal tolerance value
    c = [0.052]
    tol = [i for i in np.arange(0.00001, 0.001, 0.00001)]
    intercept_scaling = [2]
    parameters = \
            (c,
            tol,
            intercept_scaling)

    linear_svc.find_optimal_parameters(parameters)


    para = (0.052, 1e-4, 2.0)
    linear_svc.create_and_save_model(para)


if __name__ == '__main__':
    main()
