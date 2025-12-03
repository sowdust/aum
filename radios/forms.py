from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import RadioUser


class UserRegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = RadioUser
        fields = ["username", "email", "password1", "password2", "phone"]

