import json
import openai
import os
import tqdm
from time import sleep
import pandas as pd

# Set up OpenAI API key
openai.api_key = "INSERT_API_KEY_HERE"

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
request_counter = 0

temp_list = []
for i, track in enumerate(tracks):
    # Generate embeddings

    # upper limit of each genre is 8
    # so from all tracks retrieve 8 tracks from each genre
    # this is done to stay within the rate limit of openai api
    
    if genre_count[track["genre"]] < 50:
        
        request_counter +=1
        if request_counter == 60:
            request_counter = 0
            sleep(60)
        # increase current count of the genre
        genre_count[track["genre"]] += 1

        response = openai.Embedding.create(
            model=model_engine,
            input=json.dumps(track)
        )
        # Extract embeddings from response
        embedding = response["data"][0]["embedding"]

        # add the embeddings to the tracks array
        tracks[i]['embedding'] = embedding
        temp_list.append(tracks[i])
    else:
        continue

with open('tracks_with_embeddings.json', 'w') as f:
    json.dump(temp_list, f)

data = pd.read_json("tracks_with_embeddings.json")
data.to_parquet("tracks.parquet") 