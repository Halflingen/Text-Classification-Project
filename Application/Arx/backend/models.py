from django.db import models
from django.contrib.postgres.fields import ArrayField

class Article_info(models.Model):
    article_id = models.TextField(primary_key=True)
    url = models.TextField()
    category = models.CharField(max_length=20, blank=True, null=True)
    sub_category = models.CharField(max_length=20, blank=True, null=True)
    tags = ArrayField(models.CharField(max_length=50, blank=True, null=True), blank=True, null=True)
    source = models.CharField(max_length=20, blank=True, null=True)
    published = models.DateTimeField(blank=True, null=True)
    headline = models.TextField()
    classified = models.BooleanField(default=False)


    def find_football(self):
        if self.category == 'Fotball':
            return True
        else:
            return False

    def __str__(self):
        return self.article_id

    class Meta:
        db_table = 'article_info'


class Article_content(models.Model):
    article_id = models.TextField()
    content_order = models.IntegerField()
    content = models.TextField()
    html_type = models.CharField(max_length=10)
    class_field = models.CharField(max_length=15, db_column='class', blank=True, null=True)
    class_conflict = models.CharField(max_length=15, blank=True, null=True)

    def __str__(self):
        return self.article_id

    class Meta:
        db_table = 'article_content'
        unique_together = (('article_id', 'content_order'),)
