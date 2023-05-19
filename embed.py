import json
import openai
import os
import tqdm
from time import sleep

# Set up OpenAI API key
openai.api_key = "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"

# Define model and parameters
model_engine = "text-embedding-ada-002"

with open('tracks.json') as f:
    tracks = json.load(f)

# to track track count for each genre

genre_count = {
    "pop": 0,
    "country": 0,
    "blues": 0,
    "jazz": 0,
    "reggae": 0,
    "rock": 0,
    "hip hop": 0
}

for i, track in enumerate(tracks):
    # Generate embeddings

    # upper limit of each genre is 8
    # so from all tracks retrieve 8 tracks from each genre
    # this is done to stay within the rate limit of openai api
    if genre_count[track["genre"]] < 8:

        # increase current count of the genre
        genre_count[track["genre"]] += 1

        response = openai.Embedding.create(
            model=model_engine,
            input=track['lyrics']
        )
        # Extract embeddings from response
        embedding = response["data"][0]["embedding"]

        # add the embeddings to the tracks array
        tracks[i]['embedding'] = embedding
    else:
        continue

with open('tracks_with_embeddings.json', 'w') as f:
    json.dump(tracks, f)
