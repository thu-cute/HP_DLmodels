

# Create your models here.
from django.db import models

class Calculation(models.Model):
    OPERATION_CHOICES = [
        ('Summation', 'Cộng'),
        ('Subtraction', 'Trừ'),
        ('Product', 'Nhân'),
    ]
    ts1 = models.FloatField()
    ts2 = models.FloatField()
    ts3 = models.FloatField()
    ts4 = models.FloatField()
    operation = models.CharField(max_length=20, choices=OPERATION_CHOICES)
    result = models.FloatField()
    def __str__(self):
        return f"{self.ts1} {self.operation} {self.ts2} {self.ts3} {self.ts4} = {self.result}"