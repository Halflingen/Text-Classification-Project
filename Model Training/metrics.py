from sklearn.metrics import f1_score, accuracy_score, log_loss, classification_report,\
matthews_corrcoef, precision_score, recall_score, confusion_matrix
import numpy as np

import database_api

QUARIES = {}
QUARIES['cnn_store'] =\
        """
        INSERT INTO cnn1Parameters(id, embedding_dim, filter_size, kernel_size,
        pool_size, dropout_rate, dense_layer, optimizer, epoch, batch_size, dataset_version)
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
QUARIES['cnn_exists'] =\
        """
        select exists(select from cnn1parameters where embedding_dim=%s and
        filter_size=%s and kernel_size=%s and pool_size=%s and dropout_rate=%s::real and
        dense_layer=%s and optimizer=%s and epoch=%s and batch_size=%s and dataset_version=%s)
        """
QUARIES['rnn_store'] =\
        """
        INSERT INTO rnn1Parameters(id, embedding_dim, LSTM_neurons, dropout_rate,
        dropout_location, optimizer, epoch, batch_size, dataset_version)
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
QUARIES['rnn_exists'] =\
        """
        select exists(select from rnn1parameters where embedding_dim=%s and
        lstm_neurons=%s and dropout_rate=%s::real and dropout_location=%s and
        optimizer=%s and epoch=%s and batch_size=%s and dataset_version=%s)
        """
QUARIES['svc_store'] =\
        """
        INSERT INTO SVCParameters(id, vectorizer, c, gamma, shrinking, coef0, tol,
        kernel, degree, dataset_version)
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
QUARIES['svc_exists'] =\
        """
        select exists(select from svcparameters where vectorizer=%s and
        c=%s::real and gamma=%s::real and shrinking=%s and coef0=%s::real
        and tol=%s::real and kernel=%s and degree=%s and dataset_version=%s)
        """
QUARIES['linear_svc_store'] =\
        """
        INSERT INTO linearSVCParameters(id, vectorizer, c, tol, intercept_scaling,
        penalty, loss, dual, multi_class, fit_intercept, dataset_version)
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
QUARIES['linear_svc_exists'] =\
        """
        select exists(select from linearsvcparameters where vectorizer=%s and
        c=%s::real and tol=%s::real and intercept_scaling=%s::real and penalty=%s and loss=%s and dual=%s and
        multi_class=%s and fit_intercept=%s and dataset_version=%s)
        """
QUARIES['nb_store'] =\
        """
        INSERT INTO nbParameters(id, vectorizer, dist_type, alpha, dataset_version)
        VALUES(%s, %s, %s, %s, %s)
        """
QUARIES['nb_exists'] =\
        """
        select exists(select from nbparameters where vectorizer=%s and
        dist_type=%s and alpha=%s::real and dataset_version=%s)
        """


class MetricsAndParameters:
    def __init__(self, model, parameters):
        self.precision = np.zeros(10)
        self.recall = np.zeros(10)
        self.acc = np.zeros(10)
        self.f1 = np.zeros(10)
        self.mcc = np.zeros(10)
        self.c0_f1 = np.zeros(10)
        self.c1_f1 = np.zeros(10)
        self.c2_f1 = np.zeros(10)
        self.c3_f1 = np.zeros(10)
        self.c4_f1 = np.zeros(10)
        self.train_t = np.zeros(10)
        self.prediction_t = np.zeros(10)

        self.model = model
        self.parameters = parameters

        self.id_ = -1
        self.counter = 0


    def calculate_metrics(self, test_target, test_results, train_time, prediction_time):
        '''
        this function calculates the metric for each fold in the cross-validation
        training
        '''
        i = self.counter
        self.precision[i] = precision_score(test_target, test_results, average='weighted')
        self.recall[i] = recall_score(test_target, test_results, average='weighted')
        self.acc[i] = accuracy_score(test_target, test_results)
        self.f1[i] = f1_score(test_target, test_results, average='macro')
        self.mcc[i] = matthews_corrcoef(test_target, test_results)
        report = classification_report(test_target, test_results, output_dict=True)
        self.c0_f1[i] = report['0']['f1-score']
        self.c1_f1[i] = report['1']['f1-score']
        self.c2_f1[i] = report['2']['f1-score']
        self.c3_f1[i] = report['3']['f1-score']
        self.c4_f1[i] = report['4']['f1-score']
        self.train_t[i] = train_time
        self.prediction_t[i] = prediction_time
        self.counter += 1


    def store_metrics(self):
        '''
        This fuction stores the mean value of all the metric in the database
        '''
        precision = self.precision.mean()
        recall = self.recall.mean()
        acc = self.acc.mean()
        f1 = self.f1.mean()
        mcc = self.mcc.mean()
        c0_f1 = self.c0_f1.mean()
        c1_f1 = self.c1_f1.mean()
        c2_f1 = self.c2_f1.mean()
        c3_f1 = self.c3_f1.mean()
        c4_f1 = self.c4_f1.mean()
        train_time = self.train_t.mean()
        prediction_time = self.prediction_t.mean()

        data = (self.model, acc, recall, precision, f1, mcc, c0_f1, c1_f1, c2_f1, c3_f1,
                c4_f1, train_time, prediction_time)
        print(data)

        self.id_ = database_api.store_result(data)


    def store_parameters(self):
        '''
        this function stores the parameters that where used to calculate the
        metrics. And uses the same id that was generated when storing the
        metrics in the database.
        '''
        data = (self.id_, *self.parameters, 'master_arx_13_12_18')
        quary = QUARIES[self.model+'_store']
        database_api.store_parameters(quary, data)


    def duplicate(self):
        '''
        This function check if the parameters used to train the model has
        been used before
        '''
        data = (*self.parameters, 'master_arx_13_12_18')
        quary = QUARIES[self.model+'_exists']
        if(database_api.check_if_exists(quary, data)):
            return True

        return False
