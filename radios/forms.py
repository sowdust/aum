from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import RadioUser, RadioMembership, Stream, GlobalPipelineSettings


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
            "motto", "show_archive",
            "timezone", "recording_start_hour", "recording_end_hour",
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


class StreamVisibilityForm(forms.ModelForm):
    """Owner-editable: stream active toggle and visibility per stage."""
    class Meta:
        model = Stream
        fields = [
            "is_active",
            "recording_owner_visible", "recording_public_visible",
            "segmentation_owner_visible", "segmentation_public_visible",
            "fingerprinting_owner_visible", "fingerprinting_public_visible",
            "transcription_owner_visible", "transcription_public_visible",
            "summarization_owner_visible", "summarization_public_visible",
            "daily_summarization_owner_visible", "daily_summarization_public_visible",
        ]


class StreamPipelineForm(forms.ModelForm):
    """Admin-only: enable/disable each pipeline stage for a stream."""
    class Meta:
        model = Stream
        fields = [
            "enable_recording", "enable_segmentation", "enable_fingerprinting",
            "enable_transcription", "enable_summarization", "enable_daily_summarization",
        ]


class GlobalPipelineSettingsForm(forms.ModelForm):
    """Admin-only: global pipeline stage kill switches."""
    class Meta:
        model = GlobalPipelineSettings
        fields = [
            "enable_recording", "enable_segmentation", "enable_fingerprinting",
            "enable_transcription", "enable_summarization", "enable_daily_summarization",
        ]
