"""
WSGI config for arx project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

#import sys
#sys.path.append('/home/aanund/Documents/master/code/project/arx')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'arx.settings')

application = get_wsgi_application()
