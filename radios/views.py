from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from django.template import loader
from .models import Radio, Recording, AudioStream


def radios_list(request):
    radios = Radio.objects.all().values()
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
    streams = AudioStream.objects.filter(radio=radio)
    recordings = Recording.objects.filter(stream__in=streams)
    context = {
        'radio': radio,
        'recordings': recordings,
    }
    return render(request,'radio_recordings.html', context)
