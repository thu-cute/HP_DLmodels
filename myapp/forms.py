from django import forms
from .models import Calculation

class CalculationForm(forms.Form):
    ts1 = forms.FloatField(label="TS1")
    ts2 = forms.FloatField(label="TS2")
    ts3 = forms.FloatField(label="TS3")
    ts4 = forms.FloatField(label="TS4")
    class Meta:
        model = Calculation
        fields = ['ts1', 'ts2', 'ts3', 'ts4', 'operation']
    OPERATION_CHOICES = [
        ("---","---"),
        ("Cộng", "Cộng"),
        ("Trừ", "Trừ"),
        ("Nhân", "Nhân"),
    ]
    operation = forms.ChoiceField(choices=OPERATION_CHOICES, label="Loại phép tính")

