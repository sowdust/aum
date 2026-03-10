"""
Visibility enforcement for search API endpoints.

Single enforcement point: every search view calls get_visible_stream_ids()
to scope results to what the requesting user is allowed to see.
"""
from radios.models import Stream, RadioMembership


def get_visible_stream_ids(user, stage):
    """
    Return a set of Stream IDs whose results for `stage` are visible to `user`.

    Tiers:
      - Admin (is_staff): sees everything
      - Owner (RadioMembership role=owner): sees their radio's data
        where {stage}_owner_visible=True
      - Public (anonymous / non-owner): sees data where
        {stage}_public_visible=True
    """
    owner_field = f"{stage}_owner_visible"
    public_field = f"{stage}_public_visible"

    if user and user.is_staff:
        return set(Stream.objects.values_list("id", flat=True))

    # Streams visible to the public
    public_qs = Stream.objects.filter(**{public_field: True})
    visible_ids = set(public_qs.values_list("id", flat=True))

    # Add streams owned by the authenticated user (where owner_visible=True)
    if user and user.is_authenticated:
        owned_radio_slugs = RadioMembership.objects.filter(
            user=user, role="owner", verified=True
        ).values_list("radio_id", flat=True)

        owner_qs = Stream.objects.filter(
            radio_id__in=owned_radio_slugs,
            **{owner_field: True},
        )
        visible_ids.update(owner_qs.values_list("id", flat=True))

    return visible_ids
