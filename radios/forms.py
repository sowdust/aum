from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import RadioUser, RadioMembership


class UserRegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = RadioUser
        fields = ["username", "email", "password1", "password2", "phone"]


from django.forms import inlineformset_factory
from .models import Radio, Stream

class RadioCreateForm(forms.ModelForm):
    declare_owner = forms.BooleanField(
        required=False,
        label="I am responsible for this radio",
    )

    class Meta:
        model = Radio
        fields = [
            "name", "frequencies", "website", "country", "city",
            "languages", "description", "logo", "since", "until",
            "is_fm", "is_dab", "is_am", "is_sw", "is_web",
            "motto", "show_archive"
        ]

StreamFormSet = inlineformset_factory(
    Radio,
    Stream,
    fields=["name", "url", "name"],
    extra=1,
    can_delete=True,
)



class RadioMembershipChoiceForm(forms.Form):
    membership_type = forms.ChoiceField(
        choices=RadioMembership.ROLE_CHOICES,
        widget=forms.Select(attrs={"class": "select select-bordered w-full"})
    )
