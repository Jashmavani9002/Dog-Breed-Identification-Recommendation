# models.py

from django.db import models

class Dog(models.Model):
    name = models.CharField(max_length=255)
    size = models.CharField(max_length=255)
    exercise = models.CharField(max_length=255)
    size_of_home = models.CharField(max_length=255)
    grooming = models.CharField(max_length=255)
    coat_length = models.CharField(max_length=255)
    sheds = models.CharField(max_length=255)
    lifespan = models.CharField(max_length=255)
    vulnerable_native_breed = models.CharField(max_length=255)
    town_or_country = models.CharField(max_length=255)
    size_of_garden = models.CharField(max_length=255)
