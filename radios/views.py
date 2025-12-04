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


from django.core.exceptions import PermissionDenied
def edit_radio(request, slug):
    radio = get_object_or_404(Radio, slug=slug)
    if request.user not in radio.members.all():
        raise PermissionDenied()

def radios_list(request):
    radios = Radio.objects.all()
    context = {
        'radios': radios,
    }
    return render(request, 'radios_list.html', context)

def radio_detail(request, slug):
    radio = get_object_or_404(Radio, slug=slug)
    context = {
        'radio': radio,
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
