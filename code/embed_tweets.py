import os
import httpx
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI

"""
IF INVESTIGATING THIS FILE READ THIS FIRST

The vectors have already been created and are stored in the data/vector_embeddings/open_ai directory.
This script is here for reference and can be used to create new vectors if needed.
But since the vectors are already created, this script should have to be run again.
Because it costs money to create vectors. A very small amount less than a dollar but repeated runs will add up.

If you do want to run this script, make sure you have a valid OPENAI_API_KEY in your .env file.

"""

# Paths are relative to the project root, resolved from this file's location
BASE_DIR    = Path(__file__).parent.parent
INPUT_PATH  = BASE_DIR / 'data' / 'cleaned' / 'pipeline_output.csv'
OUTPUT_PATH = BASE_DIR / 'data' / 'vector_embeddings' / 'open_ai'


def embed_tweets(model: str = 'text-embedding-3-small'):
    """
    Embeds the cleanText column from INPUT_PATH using the given OpenAI model
    and saves a compressed .npz archive to OUTPUT_PATH/<model>.npz.

    Arrays are index-aligned: embeddings[i] belongs to row_ids[i] / tweet_ids[i].

    Load example:
        data       = np.load('text-embedding-3-small.npz', allow_pickle=False)
        embeddings = data['embeddings']   # (N, D) float32
        row_ids    = data['row_ids']      # join key back to pipeline_output.csv
    """
    # http_client kwarg works around openai/httpx version incompatibility
    client = OpenAI(http_client=httpx.Client())

    df    = pd.read_csv(INPUT_PATH)
    texts = df['cleanText'].fillna('').tolist()
    print(f"Embedding {len(texts)} tweets with model '{model}' ...")

    BATCH_SIZE     = 256
    all_embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch    = texts[start: start + BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in response.data])
        print(f"  {min(start + BATCH_SIZE, len(texts))} / {len(texts)}")

    embeddings_array = np.array(all_embeddings, dtype=np.float32)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_PATH / f'{model}.npz'
    np.savez_compressed(
        out_file,
        embeddings=embeddings_array,
        row_ids   =df['row_id'].to_numpy(dtype=np.int64),
        tweet_ids =df['tweet_id'].to_numpy(dtype=np.int64),
        timestamps=df['tweet_timestamp'].to_numpy(dtype=str),
    )
    size_mb = out_file.stat().st_size / 1e6
    print(f"Saved {len(all_embeddings)} embeddings ({embeddings_array.shape[1]}d) "
          f"→ '{out_file}' ({size_mb:.1f} MB)")


if __name__ == '__main__':
    embed_tweets()