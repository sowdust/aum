from django.contrib.auth import logout
from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login
from .models import EmailVerificationToken
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import reverse
from django.core.mail import send_mail
from django.utils.crypto import get_random_string
from django.conf import settings
from .models import EmailVerificationToken
from .forms import UserRegisterForm
from .models import Radio, Recording, Stream


from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from .models import RadioMembership, GlobalPipelineSettings, PIPELINE_STAGES
from .forms import StreamVisibilityForm, StreamPipelineForm, GlobalPipelineSettingsForm

def edit_radio(request, slug):
    radio = get_object_or_404(Radio, slug=slug)
    if request.user not in radio.members.all():
        raise PermissionDenied()


@login_required
def stream_settings(request, radio_slug, stream_id):
    radio = get_object_or_404(Radio, slug=radio_slug)
    stream = get_object_or_404(Stream, id=stream_id, radio=radio)

    is_owner = RadioMembership.objects.filter(
        user=request.user, radio=radio, role="owner"
    ).exists()
    is_admin = request.user.is_staff

    if not (is_owner or is_admin):
        raise PermissionDenied()

    visibility_form = StreamVisibilityForm(instance=stream)
    pipeline_form = StreamPipelineForm(instance=stream) if is_admin else None

    if request.method == "POST":
        if "save_visibility" in request.POST and (is_owner or is_admin):
            visibility_form = StreamVisibilityForm(request.POST, instance=stream)
            if visibility_form.is_valid():
                visibility_form.save()
                messages.success(request, "Visibility settings saved.")
                return redirect("stream_settings", radio_slug=radio_slug, stream_id=stream_id)
        elif "save_pipeline" in request.POST and is_admin:
            pipeline_form = StreamPipelineForm(request.POST, instance=stream)
            if pipeline_form.is_valid():
                pipeline_form.save()
                messages.success(request, "Pipeline settings saved.")
                return redirect("stream_settings", radio_slug=radio_slug, stream_id=stream_id)

    stage_statuses = {stage: stream.is_stage_active(stage) for stage in PIPELINE_STAGES}

    return render(request, "stream_settings.html", {
        "radio": radio,
        "stream": stream,
        "visibility_form": visibility_form,
        "pipeline_form": pipeline_form,
        "stage_statuses": stage_statuses,
        "is_owner": is_owner,
        "is_admin": is_admin,
        "pipeline_stages": PIPELINE_STAGES,
    })


@login_required
def global_pipeline_settings(request):
    if not request.user.is_staff:
        raise PermissionDenied()

    settings_obj = GlobalPipelineSettings.get_settings()
    form = GlobalPipelineSettingsForm(instance=settings_obj)

    if request.method == "POST":
        form = GlobalPipelineSettingsForm(request.POST, instance=settings_obj)
        if form.is_valid():
            form.save()
            messages.success(request, "Global pipeline settings saved.")
            return redirect("global_pipeline_settings")

    return render(request, "global_pipeline_settings.html", {
        "form": form,
        "pipeline_stages": PIPELINE_STAGES,
    })

def radios_list(request):
    radios = Radio.objects.all()
    context = {
        'radios': radios,
    }
    return render(request, 'radios_list.html', context)

def radio_detail(request, slug):
    radio = get_object_or_404(Radio, slug=slug)
    is_owner = (
        request.user.is_authenticated
        and RadioMembership.objects.filter(user=request.user, radio=radio, role="owner").exists()
    )
    is_admin = request.user.is_authenticated and request.user.is_staff
    context = {
        'radio': radio,
        'is_owner': is_owner,   # True if the logged-in user owns this radio
        'is_admin': is_admin,   # True if the logged-in user is site staff
    }
    return render(request, 'radio_detail.html', context)

def radio_recordings(request, slug):
    radio = get_object_or_404(Radio, slug=slug)
    streams = Stream.objects.filter(radio=radio)
    recordings = Recording.objects.filter(stream__in=streams)
    start = request.GET.get("start")
    end = request.GET.get("end")
    if start:
        recordings = recordings.filter(start_time__gte=start)
    if end:
        recordings = recordings.filter(start_time__lte=end)

    recordings = recordings.order_by("-start_time")

    context = {
        'radio': radio,
        'recordings': recordings,
        'start': start,
        'end' : end
    }
    return render(request,'radio_recordings.html', context)

def register(request):
    if request.method == "POST":
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False
            user.save()
            token = get_random_string(48)
            EmailVerificationToken.objects.create(user=user, token=token)
            verify_url = request.build_absolute_uri(reverse("verify_email", args=[token]))
            send_mail(
            "Verify your account",
            f"Click to verify: {verify_url}",
            settings.DEFAULT_FROM_EMAIL,
            [user.email],
            fail_silently=False,
            )

            return render(request, "accounts/verify_sent.html")

    else:
        form = UserRegisterForm()

    return render(request, "accounts/register.html", {"form": form})

def verify_email(request, token):
    token_obj = get_object_or_404(EmailVerificationToken, token=token)

    user = token_obj.user
    user.is_active = True
    user.save()

    token_obj.delete()

    login(request, user)  # optional: log them in automatically

    return render(request, "accounts/verify_success.html", {"user": user})



def logout_view(request):
    logout(request)
    return redirect("login")

"""
# views.py
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from .forms import RadioCreateForm, StreamFormSet
from .models import RadioMembership

@login_required
def radio_create(request):
    if request.method == "POST":
        form = RadioCreateForm(request.POST, request.FILES)
        formset = StreamFormSet(request.POST)

        if form.is_valid() and formset.is_valid():
            radio = form.save()

            # Assign ownership if checkbox checked
            if form.cleaned_data.get("declare_owner"):
                RadioMembership.objects.create(
                    user=request.user,
                    radio=radio,
                    role="owner",
                    verified="false",
                )
            else:
                # Otherwise, user becomes simple member (optional)
                RadioMembership.objects.create(
                    user=request.user,
                    radio=radio,
                    role="member",
                )

            formset.instance = radio
            formset.save()

            return redirect("radio_detail", pk=radio.slug)

    else:
        form = RadioCreateForm()
        formset = StreamFormSet()

    return render(request, "radios/radio_create.html", {
        "form": form,
        "formset": formset,
    })




# radios/views.py



"""

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import RadioCreateForm, StreamFormSet, RadioMembershipChoiceForm
from .models import Radio, RadioMembership

@login_required
def radio_create(request):
    if request.method == "POST":
        form = RadioCreateForm(request.POST, request.FILES)
        membership_form = RadioMembershipChoiceForm(request.POST)
        formset = StreamFormSet(request.POST)

        if form.is_valid() and formset.is_valid() and membership_form.is_valid():
            radio = form.save()

            # Save streams
            formset.instance = radio
            formset.save()

            # Save membership with selected role
            RadioMembership.objects.create(
                user=request.user,
                radio=radio,
                role=membership_form.cleaned_data["membership_type"]
            )

            return redirect("radio_detail", slug=radio.slug)
        else:
            print(form.is_valid())
            print(formset.is_valid())
            print(membership_form.is_valid())
            print("Form not valid!")

    else:
        form = RadioCreateForm()
        membership_form = RadioMembershipChoiceForm()
        formset = StreamFormSet()

    return render(
        request,
        "radio_create.html",
        {
            "form": form,
            "membership_form": membership_form,
            "formset": formset,
        }
    )
