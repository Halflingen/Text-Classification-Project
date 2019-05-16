from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
import time
import numpy as np
import pickle

import get_data
import metrics


class NaiveBayes:
    def __init__(self):
        self.data = get_data.Data()
        self.train_data,\
        self.train_target,\
        self.test_data,\
        self.test_target\
        = self.data.fetch_train_data()


    def create_model(self, para):
        '''
        0 - vectorizer
        1 - dist_type
        2 - alpha
        '''
        vectorizer = CountVectorizer()
        if para[0] == 'tfidf':
            vectorizer = TfidfVectorizer()

        clf = MultinomialNB(alpha=para[2])
        if para[1] == 'bernoulli':
            clf = BernoulliNB(alpha=para[2])

        pipeline = Pipeline([
            ('vec', vectorizer),
            ('clf', clf),
        ])

        return pipeline


    def train_model(self, parameters):
        '''
        0 - vectorizer
        1 - dist_type
        2 - alpha
        '''

        metric = metrics.MetricsAndParameters('nb', parameters)

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

        with open('t2_model_nb.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    nb = NaiveBayes()
    vectorizer = ['Count', 'tfidf']
    dist_type = ['bernoulli', 'multinomial']
    alpha = [i for i in np.arange(0.01, 4, 0.05)]
    parameters = \
            (vectorizer,
            dist_type,
            alpha)

    nb.find_optimal_parameters(parameters)

    #para = ('count', 'bernoulli', 1.74)
    #nb.create_and_save_model(para)


if __name__ == '__main__':
    main()
