from rest_framework import serializers
from backend.models import Article_info, Article_content

class Article_info_serializer(serializers.ModelSerializer):
    class Meta:
        model = Article_info
        fields = ('headline', 'article_id', 'source', 'category', 'sub_category', 'published', 'tags')


class ArticleContentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article_content
        fields = ('id', 'article_id', 'content_order', 'content', 'html_type', 'class_field')


class ParagraphSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article_content
        fields = ('class_conflict', 'class_field')


class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article_info
        fields = ('classified',)


class StatisticSerializer(serializers.ModelSerializer):
    total = serializers.SerializerMethodField()

    class Meta:
        model = Article_content
        fields = ('total',)

    def get_total(self, obj):
        return obj
