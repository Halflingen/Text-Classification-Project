from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
    path('article_info/', views.ListArticles.as_view()),
    path('article_info/<slug:id_>', views.Article.as_view()),
    path('article_content/<slug:id_>', views.ListArticleContent.as_view()),
    path('article_content/<slug:id_>/<int:content_order>', views.Paragraph.as_view()),
    path('fast_classify/<str:id_>', views.FastClassify.as_view()),
    path('player_articles/<str:name_tag>', views.PlayerArticles.as_view()),
    path('player_classifier/<str:model_type>/<str:name_tag>', views.PlayerClassifier.as_view()),
    path('statistic/', views.Statistic.as_view()),
]
