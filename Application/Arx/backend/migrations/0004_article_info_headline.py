# Generated by Django 2.1.1 on 2018-08-31 13:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backend', '0003_auto_20180830_1452'),
    ]

    operations = [
        migrations.AddField(
            model_name='article_info',
            name='headline',
            field=models.TextField(default='na'),
            preserve_default=False,
        ),
    ]
