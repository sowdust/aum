from django import template

register = template.Library()


@register.filter
def format_seconds(value):
    """Convert float seconds to H:MM:SS or M:SS string."""
    try:
        total = int(float(value))
    except (TypeError, ValueError):
        return value
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
