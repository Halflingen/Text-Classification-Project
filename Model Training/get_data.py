from sklearn.model_selection import train_test_split
import numpy as np
import re
import random

import database_api


class Data:
    def __init__(self):
        self.paragraph_length = 65
        rows =  database_api.get_dataset()
        random.shuffle(rows)
        self.data, self.target = self.prepear_data(rows)


    def prepear_data(self, rows):
        data, target = self.split_target_and_data(rows)
        target, data = self.change_label_name(target,data)
        data = self.remove_extra_spaces(data)
        data, target = self.remove_paragraphs_over_x_words(data, target)

        return data, target


    def split_target_and_data(self, rows):
        '''
        this functions splits data and target, into two seperate arrays
        '''
        data = []
        target = []
        for x in rows:
            data.append(x[3])
            target.append(x[5])

        return data, target


    def change_label_name(self, labels, data):
        '''
        Change label name from - to:
        Ignore - 0
        Goal/Assist - 1
        Transfer - 2
        Quote - 3
        Irrelevant - 4
        '''
        target = []
        new_data = []
        for x,y in zip(labels,data):
            if x == 'Ignore':
                target.append(0)
            elif x == 'Goal/Assist':
                target.append(1)
            elif x == 'sjanse':
                target.append(1)
            elif x == 'Transfer':
                target.append(2)
            elif x == 'quote':
                target.append(3)
            elif x == 'irrelevant':
                target.append(4)
            else:
                target.append(4)
                #continue

            #elif x == 'Club details':
                #continue
            #elif x == 'Player details':
                #continue
            new_data.append(y)

        return target, new_data


    def remove_extra_spaces(self, data):
        new_data = []
        for x in data:
            new_data.append(re.sub(' +',' ', x))

        return new_data


    def remove_paragraphs_over_x_words(self, data, target):
        new_data = []
        new_target = []
        for x,y in zip(data,target):
            if len(x.split(' ')) > self.paragraph_length:
                continue
            new_data.append(x)
            new_target.append(y)

        return new_data, new_target


    def fetch_train_data(self):
        return np.array(self.data), np.array(self.target), 0, 0


    def fetch_train_and_test_data(self, split_precentage, ml_type):
        train_data,\
        test_data,\
        train_target,\
        test_target = \
        train_test_split(self.data, self.target, test_size=split_precentage, stratify=target)
        if ml_type == "dl":
            return np.array(train_data), np.array(train_target), test_data, np.array(test_target)
        elif ml_type == "ml":
            return np.array(train_data), np.array(train_target), np.array(test_data), np.array(test_target)
