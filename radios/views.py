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
from .models import Radio, Recording, Stream, TranscriptionSegment
import os


from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.db.models import Count, Q
from django.utils import timezone
from .models import RadioMembership, GlobalPipelineSettings, PIPELINE_STAGES, TranscriptionSettings, SummarizationSettings, SongOccurrence, BroadcastDaySummary, DailySummarizationSettings
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


@login_required
def admin_dashboard(request):
    if not request.user.is_staff:
        raise PermissionDenied()

    global_settings = GlobalPipelineSettings.get_settings()
    transcription_cfg = TranscriptionSettings.get_settings()
    summarization_cfg = SummarizationSettings.get_settings()

    global_stage_flags = {
        stage: getattr(global_settings, f"enable_{stage}")
        for stage in PIPELINE_STAGES
    }

    active_stream_count = Stream.objects.filter(is_active=True).count()

    ANALYSIS_STAGES = ["segmentation", "fingerprinting", "transcription", "summarization"]

    stage_counts_list = []
    for stage in ANALYSIS_STAGES:
        status_field = f"{stage}_status"
        rows = Recording.objects.values(status_field).annotate(n=Count("id"))
        counts = {row[status_field]: row["n"] for row in rows}
        stage_counts_list.append({"stage": stage, "counts": counts})

    running_per_stage = {}
    for stage in ANALYSIS_STAGES:
        status_field = f"{stage}_status"
        qs = (
            Recording.objects
            .filter(**{status_field: "running"})
            .select_related("stream__radio")
            .order_by("analysis_started_at")[:20]
        )
        running_per_stage[stage] = list(qs)

    recent_failed_recs = (
        Recording.objects
        .filter(
            Q(segmentation_status="failed") |
            Q(fingerprinting_status="failed") |
            Q(transcription_status="failed") |
            Q(summarization_status="failed")
        )
        .select_related("stream__radio")
        .order_by("-start_time")[:50]
    )
    failure_rows = []
    for rec in recent_failed_recs:
        for stage in ANALYSIS_STAGES:
            if getattr(rec, f"{stage}_status") == "failed":
                failure_rows.append({
                    "recording_id": rec.id,
                    "stage": stage,
                    "stream_name": rec.stream.name,
                    "radio": rec.stream.radio,
                    "timestamp": rec.start_time,
                    "error_excerpt": (getattr(rec, f"{stage}_error", "") or "")[:300],
                })
    failure_rows.sort(key=lambda r: r["timestamp"] or "", reverse=True)
    failure_rows = failure_rows[:100]

    running_stages_list = [
        {"stage": stage, "jobs": running_per_stage[stage]}
        for stage in ANALYSIS_STAGES
        if running_per_stage[stage]
    ]
    any_running = bool(running_stages_list)

    # Daily summarization stats (BroadcastDaySummary is radio+date level, not per-recording)
    daily_summ_counts = {}
    for row in BroadcastDaySummary.objects.values("status").annotate(n=Count("id")):
        daily_summ_counts[row["status"]] = row["n"]

    daily_summ_running = list(
        BroadcastDaySummary.objects
        .filter(status="running")
        .select_related("radio")
        .order_by("updated_at")[:20]
    )

    daily_summ_failed = list(
        BroadcastDaySummary.objects
        .filter(status="failed")
        .select_related("radio")
        .order_by("-updated_at")[:20]
    )

    daily_summ_cfg = DailySummarizationSettings.get_settings()

    # Recording status panel
    recording_streams = (
        Stream.objects.filter(is_active=True, recording_status="recording")
        .select_related("radio", "audio_feed")
    )
    recording_error_streams = (
        Stream.objects.filter(recording_status="error")
        .select_related("radio", "audio_feed")
    )

    return render(request, "admin_dashboard.html", {
        "global_stage_flags": global_stage_flags,
        "pipeline_stages": PIPELINE_STAGES,
        "active_stream_count": active_stream_count,
        "stage_counts_list": stage_counts_list,
        "running_stages_list": running_stages_list,
        "failure_rows": failure_rows,
        "transcription_backend": transcription_cfg.get_backend_display(),
        "summarization_backend": summarization_cfg.get_backend_display(),
        "any_running": any_running,
        "recording_streams": recording_streams,
        "recording_error_streams": recording_error_streams,
        "daily_summ_counts": daily_summ_counts,
        "daily_summ_running": daily_summ_running,
        "daily_summ_failed": daily_summ_failed,
        "daily_summarization_backend": daily_summ_cfg.get_backend_display(),
        "now": timezone.now(),
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

    recordings = (
        recordings
        .select_related("stream")
        .prefetch_related(
            "segments__song_occurrences__song__artist_ref",
            "segments__song_occurrences__song__genres",
            "chunk_summary__tags",
        )
        .order_by("-start_time")
    )

    context = {
        "radio": radio,
        "recordings": recordings,
        "start": start,
        "end": end,
    }
    return render(request, "radio_recordings.html", context)

def radio_segments(request, slug):
    """Browse segments for a radio, with filtering by time and type."""
    from django.core.paginator import Paginator

    radio = get_object_or_404(Radio, slug=slug)
    streams = Stream.objects.filter(radio=radio)

    qs = (
        TranscriptionSegment.objects
        .filter(recording__stream__in=streams)
        .select_related("recording__stream")
        .prefetch_related(
            "song_occurrences__song__artist_ref",
            "song_occurrences__song__genres",
        )
    )

    start = request.GET.get("start")
    end = request.GET.get("end")
    seg_type = request.GET.get("type", "").strip()

    if start:
        qs = qs.filter(
            Q(absolute_start_time__gte=start) |
            Q(absolute_start_time__isnull=True, recording__start_time__gte=start)
        )
    if end:
        qs = qs.filter(
            Q(absolute_end_time__lte=end) |
            Q(absolute_end_time__isnull=True, recording__start_time__lte=end)
        )
    if seg_type:
        qs = qs.filter(segment_type=seg_type)

    qs = qs.order_by("-recording__start_time", "start_offset")

    paginator = Paginator(qs, 50)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    # Build query string without 'page' for pagination links
    params = request.GET.copy()
    params.pop("page", None)
    page_qs = params.urlencode()

    context = {
        "radio": radio,
        "segments": page_obj,
        "start": start,
        "end": end,
        "seg_type": seg_type,
        "page_qs": page_qs,
    }
    return render(request, "radio_segments.html", context)


def songs_played(request):
    """
    List all identified song occurrences, newest first.
    Respects fingerprinting visibility settings.
    Supports optional filtering by radio slug and date range.
    """
    qs = (
        SongOccurrence.objects
        .select_related(
            "song", "song__artist_ref",
            "segment__recording__stream__radio",
        )
        .prefetch_related("song__genres")
        .order_by("-segment__recording__start_time", "start_offset")
    )

    # Visibility: only show occurrences from publicly-visible streams,
    # plus owner-visible streams if the user is authenticated.
    if request.user.is_staff:
        pass  # staff see everything
    else:
        from radios.api.permissions import get_visible_stream_ids
        visible = get_visible_stream_ids(request.user, "fingerprinting")
        qs = qs.filter(segment__recording__stream_id__in=visible)

    # Optional filters
    radio_slug = request.GET.get("radio", "").strip()
    if radio_slug:
        qs = qs.filter(segment__recording__stream__radio__slug=radio_slug)

    date_from = request.GET.get("date_from", "").strip()
    if date_from:
        qs = qs.filter(segment__recording__start_time__date__gte=date_from)

    date_to = request.GET.get("date_to", "").strip()
    if date_to:
        qs = qs.filter(segment__recording__start_time__date__lte=date_to)

    radios = Radio.objects.order_by("name")

    context = {
        "occurrences": qs[:200],  # cap at 200 rows; paginate later if needed
        "radios": radios,
        "radio_slug": radio_slug,
        "date_from": date_from,
        "date_to": date_to,
    }
    return render(request, "songs_played.html", context)


@login_required
def recording_delete(request, slug, recording_id):
    if not request.user.is_staff:
        raise PermissionDenied()

    radio = get_object_or_404(Radio, slug=slug)
    # Scope the lookup to recordings belonging to this radio to prevent IDOR
    recording = get_object_or_404(
        Recording, id=recording_id, stream__radio=radio
    )

    if request.method == "POST":
        file_path = recording.file.path if recording.file else None
        recording.delete()  # cascades to segments, summaries, etc.
        if file_path and os.path.isfile(file_path):
            os.remove(file_path)
        messages.success(request, "Recording deleted.")
        return redirect("radio_recordings", slug=slug)

    # GET: show confirmation page
    return render(request, "recording_confirm_delete.html", {
        "radio": radio,
        "recording": recording,
    })


def broadcast_day_list(request, slug):
    """List available broadcast day summaries for a radio."""
    radio = get_object_or_404(Radio, slug=slug)

    qs = BroadcastDaySummary.objects.filter(radio=radio, status="done")

    # Visibility check
    if not (request.user.is_authenticated and request.user.is_staff):
        is_owner = (
            request.user.is_authenticated
            and RadioMembership.objects.filter(
                user=request.user, radio=radio, role="owner"
            ).exists()
        )
        if is_owner:
            # Show if any stream has daily_summarization_owner_visible
            has_visible = radio.streams.filter(
                daily_summarization_owner_visible=True
            ).exists()
            if not has_visible:
                qs = qs.none()
        else:
            # Public: check daily_summarization_public_visible
            has_visible = radio.streams.filter(
                daily_summarization_public_visible=True
            ).exists()
            if not has_visible:
                qs = qs.none()

    return render(request, "broadcast_day_list.html", {
        "radio": radio,
        "broadcast_days": qs,
    })


def broadcast_day(request, slug, date):
    """Display a structured broadcast day summary."""
    from datetime import date as date_type

    radio = get_object_or_404(Radio, slug=slug)

    try:
        day = date_type.fromisoformat(date)
    except ValueError:
        from django.http import Http404
        raise Http404("Invalid date format. Use YYYY-MM-DD.")

    summary = get_object_or_404(
        BroadcastDaySummary, radio=radio, date=day, status="done"
    )

    # Visibility check
    if not (request.user.is_authenticated and request.user.is_staff):
        is_owner = (
            request.user.is_authenticated
            and RadioMembership.objects.filter(
                user=request.user, radio=radio, role="owner"
            ).exists()
        )
        if is_owner:
            has_visible = radio.streams.filter(
                daily_summarization_owner_visible=True
            ).exists()
        else:
            has_visible = radio.streams.filter(
                daily_summarization_public_visible=True
            ).exists()

        if not has_visible:
            from django.core.exceptions import PermissionDenied
            raise PermissionDenied()

    # Prefetch related data
    shows = (
        summary.shows
        .prefetch_related(
            "tags",
            "songs__song__artist_ref",
            "songs__song__genres",
            "songs__segment__recording",
        )
        .order_by("order", "start_time")
    )

    return render(request, "broadcast_day.html", {
        "radio": radio,
        "summary": summary,
        "shows": shows,
    })


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
