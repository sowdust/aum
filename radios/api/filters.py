import django_filters
from django.db.models import Q

from radios.models import Radio


BROADCAST_TYPE_MAP = {
    "fm": "is_fm",
    "am": "is_am",
    "dab": "is_dab",
    "sw": "is_sw",
    "web": "is_web",
}


class RadioFilter(django_filters.FilterSet):
    q = django_filters.CharFilter(method="filter_q", label="Search")
    country = django_filters.CharFilter(field_name="country", lookup_expr="exact")
    city = django_filters.CharFilter(field_name="city", lookup_expr="icontains")
    broadcast_type = django_filters.CharFilter(method="filter_broadcast_type", label="Broadcast type")
    language = django_filters.CharFilter(field_name="languages", lookup_expr="icontains")

    class Meta:
        model = Radio
        fields = []

    def filter_q(self, queryset, name, value):
        if not value:
            return queryset
        return queryset.filter(
            Q(name__icontains=value)
            | Q(description__icontains=value)
            | Q(city__icontains=value)
            | Q(motto__icontains=value)
            | Q(frequencies__icontains=value)
        )

    def filter_broadcast_type(self, queryset, name, value):
        field = BROADCAST_TYPE_MAP.get(value.lower())
        if field:
            return queryset.filter(**{field: True})
        return queryset
