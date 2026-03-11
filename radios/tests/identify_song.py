import acoustid
import sys

def identify_song(api_key, file_path):
    results = acoustid.match(api_key, file_path)

    for score, recording_id, title, artist in results:
        print(f"Title: {title}")
        print(f"Artist: {artist}")
        print(f"Score: {score}")
        print(f"Recording ID: {recording_id}")
        print("-" * 40)

if __name__ == "__main__":
    api_key = sys.argv[1]
    file_path = sys.argv[2]

    identify_song(api_key, file_path)