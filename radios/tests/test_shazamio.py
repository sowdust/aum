import asyncio
from shazamio import Shazam

async def main():
    s = Shazam()
    # Use any known Shazam track key (e.g. from a previous fingerprint)
    r = await s.track_about(549952578)
    # Look for duration-related keys
    for section in r.get('sections', []):
        if 'metadata' in section:
            for m in section['metadata']:
                print(m.get('title'), ':', m.get('text'))
    # Also check top-level keys
    for k in r.keys():
        print(f'top-level key: {k}')
asyncio.run(main())
