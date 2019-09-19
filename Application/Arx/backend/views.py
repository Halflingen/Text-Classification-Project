from backend.models import Article_info, Article_content
from backend.serializers import Article_info_serializer, ArticleContentSerializer,\
ParagraphSerializer, ArticleSerializer, StatisticSerializer
from rest_framework import generics
from django.db.models import Count

from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.mixins import LoginRequiredMixin

from django.http import JsonResponse
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, log_loss, classification_report,\
matthews_corrcoef, precision_score, recall_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels

import os
from django.conf import settings

import random
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf
from keras import backend as K


class ListArticles(APIView):
    '''
    This view returns 50 random article titles, with other information.
    Articles that is marked as classified is ignored
    '''
    authentication_classes = (SessionAuthentication, BasicAuthentication)
    permission_classes = (IsAuthenticated,)

    def get(self, request, format=None):
        articles = Article_info.objects.exclude(headline='na').exclude(classified=True)\
        .order_by('?')[:50]
        serializer = Article_info_serializer(articles, many=True)
        return Response(serializer.data)


class Article(APIView):
    '''
    This view takes an artice_id as input and setts the corresponding article
    to classified. Articles labled classified will not be returned by
    ListArticles
    '''
    authentication_classes = (SessionAuthentication, BasicAuthentication)
    permission_classes = (IsAuthenticated,)

    def put(self, request, id_, format=None):
        obj = Article_info.objects.get(article_id=id_)
        serializer = ArticleSerializer(obj, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(status=status.HTTP_204_NO_CONTENT)


class ListArticleContent(APIView):
    '''
    this view takes an article_id as input and returns all the content assosieted
    with the article.
    The content is ordered as displayed in the original article
    '''
    authentication_classes = (SessionAuthentication, BasicAuthentication)
    permission_classes = (IsAuthenticated,)

    def get(self, request, id_, format=None):
        article = Article_content.objects.filter(article_id=id_).order_by('content_order')
        serializer = ArticleContentSerializer(article, many=True)
        return Response(serializer.data)


class Paragraph(APIView):
    '''
    This view takes an article_id and content oreder as input and labels the
    corresponding paragraph with the class label found in the body.
    It also sets the user how did the labeling.
    '''
    authentication_classes = (SessionAuthentication, BasicAuthentication)
    permission_classes = (IsAuthenticated,)

    def put(self, request, id_, content_order, format=None):
        data = {'class_conflict': str(request.user), 'class_field':request.data['class_field']}
        obj = Article_content.objects.get(article_id=id_, content_order=content_order)
        serializer = ParagraphSerializer(obj, data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(status=status.HTTP_204_NO_CONTENT)


class FastClassify(APIView):
    '''
    This view takes a name tag as input and returns every paragraph containing
    the name tag in the database
    '''
    authentication_classes = (SessionAuthentication, BasicAuthentication)
    permission_classes = (IsAuthenticated,)

    def get(self, request, id_, format=None):
        article = Article_content.objects.filter(class_field=None).filter(content__search=id_)\
        .order_by('content_order')
        serializer = ArticleContentSerializer(article, many=True)
        return Response(serializer.data)


class PlayerArticles(APIView):
    authentication_classes = (SessionAuthentication, BasicAuthentication)
    permission_classes = (IsAuthenticated,)

    def get(self, request, name_tag, format=None):
        paragraphs = Article_content.objects.filter(content__search=name_tag)\
        .values('article_id').annotate(total=Count('article_id'))
        ids = []
        for x in paragraphs:
            ids.append(x['article_id'])
        articles = Article_info.objects.filter(article_id__in=ids).exclude(classified=True)
        serializer = Article_info_serializer(articles, many=True)
        return Response(serializer.data)


class Statistic(APIView):
    '''
    This view returns the number of labes labeled for each class
    '''
    authentication_classes = (SessionAuthentication, BasicAuthentication)
    permission_classes = (IsAuthenticated,)

    def get(self, request, format=None):
        stat = Article_content.objects.all().values('class_field')\
        .annotate(total=Count('class_field')).order_by()
        serializer = StatisticSerializer(stat, many=True)
        return Response(serializer.data)


##############################################################################
#                         Machine learning API                               #
##############################################################################


class PlayerClassifier(APIView):
    authentication_classes = (SessionAuthentication, BasicAuthentication)
    permission_classes = (IsAuthenticated,)

    def predict_paragraphs(self,paragraphs, class_label, model_type):
        label_num = {}
        label_num['Goal/Assist'] = '1'
        label_num['Transfer'] = '2'
        label_num['Quote'] = '3'
        label_num['quote'] = '3'
        label_num['Irrelevant'] = '4'
        label_num['irrelevant'] = '4'
        label_num['Ignore'] = '0'
        label_num['None'] = '9'
        label_num['Player details'] = '9'
        label_num['Club details'] = '9'
        label_num['sjanse'] = '9'
        label_num['Injuries'] = '9'
        label_num['Rodt/gult kort'] = '9'
        if model_type == 'cnn' or model_type == 'rnn':
            tokenizer_name = "backend/ml_models/tokenizer_"+model_type+".pickle"
            model_name = "backend/ml_models/model_"+model_type+".h5"
            tokenizer = 0
            with open(os.path.join(settings.BASE_DIR, tokenizer_name), 'rb') as handle:
                tokenizer = pickle.load(handle)
            model = load_model(os.path.join(settings.BASE_DIR, model_name))
            encoded_data = tokenizer.texts_to_sequences(paragraphs)
            padded_data = pad_sequences(encoded_data, maxlen=65, padding='post')
            score = model.predict(padded_data, batch_size=32)
            results = score.argmax(axis=-1)
            result_ = [{'content':x, 'label':str(y), 'class':label_num[z]}
                    for x,y,z in zip(paragraphs,results, class_label)]
            K.clear_session()
            return result_

        model_name = "backend/ml_models/model_"+model_type+".pickle"
        model = 0
        with open(os.path.join(settings.BASE_DIR, model_name), 'rb') as handle:
            model = pickle.load(handle)

        results = model.predict(paragraphs)

        result_ = [{'content':x, 'label':str(y), 'class':label_num[z]}
                for x,y,z in zip(paragraphs,results, class_label)]

        K.clear_session()
        return result_

    def get(self, request, model_type, name_tag, format=None):
        paragraphs = Article_content.objects.filter(content__search=name_tag)\
        .values('article_id').annotate(total=Count('article_id'))
        respons_data = {}
        ids = []
        for x in paragraphs:
            if x['total'] < 2: continue
            ids.append(x['article_id'])
        if (len(ids) == 0):
            return Response(status=status.HTTP_204_NO_CONTENT)
        paragraphs = []
        class_label = []
        for id_ in ids:
            article = Article_content.objects.filter(article_id=id_)
            serializer = ArticleContentSerializer(article, many=True)
            for para in serializer.data:
                paragraphs.append(para['content'])
                if para['class_field'] == '':
                    class_label.append('None')
                    continue
                #paragraphs.append(para['content'])
                class_label.append(para['class_field'])

        res = self.predict_paragraphs(paragraphs,class_label, model_type)
        return JsonResponse({'data':res})
