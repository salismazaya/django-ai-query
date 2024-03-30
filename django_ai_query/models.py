from django.db import models
from .queryset import AIQueryset

class AIModel(models.Model):
    class Meta:
        abstract = True

    objects = AIQueryset.as_manager()