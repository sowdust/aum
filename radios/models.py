from django.db import models

# Create your models here.
class Radio(models.Model):
	name = models.CharField(max_length=255)
	homepage = models.CharField(max_length=255)
	streaming = models.CharField(max_length=255)
	