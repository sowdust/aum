from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .models import Radio

def radios(request):
    radios = Radio.objects.all().values()
    context = {
        'radios': radios,
    }
    template = loader.get_template('index.html')
    return HttpResponse(template.render(context, request))
