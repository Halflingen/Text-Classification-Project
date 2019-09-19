#!/bin/sh
ls
python Arx/manage.py makemigrations
python Arx/manage.py migrate
python Data/add_data.py
exec "$@"
