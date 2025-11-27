from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from django.template import loader
from .models import Radio

def radios(request):
    radios = Radio.objects.all().values()
    context = {
        'radios': radios,
    }
    template = loader.get_template('index.html')
    return render(request, 'index.html', context)
