# package the semantic search into an API with front end
# set up a flask app below

import collections
import openai
import pandas as pd
import numpy as np
import sys


global FILE_NAME
FILE_NAME = 'tracks.parquet'
EMBEDDINGS_MODEL = "text-embedding-ada-002"
openai.api_key = "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"


def parse_dataset():
    """
    Parse a dataset of preprocessed text and embeddings.

    Returns:
    - An instance of the collections.namedtuple 'Engine', containing the following attributes:
        * lyrics_corpus: a list of strings representing the preprocessed text data.
        * lyrics_corpus:_embeddings: a numpy array of shape (n, m) representing the precomputed embeddings for each text data.
        * artist_name_list: a list of strings representing the artist name for each track.
        * track_name_list: a list of strings representing the track_name for each lyrics text data.
        * top_k: an integer representing the maximum number of similar sections to return for a given query.

    Example:
    >>> engine = parse_dataset()
    """

    print("Loading embedding file...")
    tgg_file = pd.read_parquet(FILE_NAME)
    print("Converting to np array...")
    artist_name_list = tgg_file['artist_name'].tolist()
    track_name_list = tgg_file['track_name'].tolist()
    lyrics_corpus = tgg_file['lyrics'].tolist()
    genre_list = tgg_file['genre'].tolist()
    lyrics_corpus_embeddings = np.array(tgg_file['embedding'].tolist(), dtype=float)
    top_k = min(3, len(lyrics_corpus))
    print("done converting to np array")
    return collections.namedtuple('Engine', 
    ['lyrics_corpus', 
    'lyrics_corpus_embeddings', 
    'artist_name_list',
    'track_name_list',
    'genre_list',
    'top_k'])(
        lyrics_corpus, 
        lyrics_corpus_embeddings, 
        artist_name_list,
        track_name_list,
        genre_list,
        top_k)

def get_query_embedding_openai(prompt):
    response = openai.Embedding.create(
        model=EMBEDDINGS_MODEL,
        input=prompt
    )
    return response['data'][0]['embedding']

def prepare_contexts(dataset):
    """
    Create a dictionary of document section embeddings.

    Args:
    - dataset: contains preprocessed text data and their embeddings.

    Returns:
    - A dictionary where each key is a tuple representing a document section consisting of (page_text, page_number, chapter_number), 
    and each value is the corresponding embedding.
    """
    contexts = {}
    for lyrics, artist_name, track_name, genre, embedding in zip(
        dataset.lyrics_corpus, 
        dataset.artist_name_list, 
        dataset.track_name_list,
        dataset.genre_list,
        dataset.lyrics_corpus_embeddings
    ):
        contexts[(lyrics, artist_name, track_name, genre)] = embedding
    return contexts

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query_embedding: str, contexts: dict[(str, int, int), np.array]) -> list[(float, (str, int, int))]:
    """
    Compare query embedding against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def get_semantic_suggestions(prompt):
    """
    Generate a list of semantic suggestions based on a given prompt.

    Args:
    - prompt: a string representing the user's query.

    Returns:
    - A list of dictionaries containing the top-k most relevant document sections to the query.
        Each dictionary has the following keys:
            * lyrics: a string representing the lyrics of a track
            * track: a string representing the name of the track.
            * artist: a string representing the artist name for each track.

    Example:
    >>> suggestions = get_semantic_suggestions("What is the meaning of life?")
    """
    dataset_with_embeddings = parse_dataset()
    query_embedding = np.array(get_query_embedding_openai(prompt), dtype=float)
    relevant_sections = order_document_sections_by_query_similarity(
        query_embedding, 
        prepare_contexts(dataset=dataset_with_embeddings)
    )
    top_three = relevant_sections[:dataset_with_embeddings.top_k]
    final = []
    for _, (lyrics, artist_name, track_name, genre) in top_three:
        final.append(
            {
                'lyrics': lyrics,
                'artist': artist_name,
                'track': track_name,
                'genre': genre
            })
    return final 



if __name__ == '__main__':
    prompt = sys.argv[1]
    results = get_semantic_suggestions(prompt)
    for result in results:
        print("-"*80)
        print(f"Track Name: {result['track']}, Artist: {result['artist']}, Genre: {result['genre']}")
        print(result['lyrics'])
