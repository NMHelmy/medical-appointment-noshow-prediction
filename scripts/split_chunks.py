# Nour Helmy - 202202012
# Split Dataset into Chunks for Fake Streaming

import pandas as pd
import os
from config import RAW_FILE, CHUNKS_DIR

CHUNK_SIZE  = 10_000   # rows per chunk -> 11 chunks from 110,527 rows

def split_into_chunks():
    # Load full dataset
    df = pd.read_csv(RAW_FILE) 
    print(f"Total rows loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Clean output directory
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(CHUNKS_DIR) if f.endswith('.csv')] # Only remove existing CSV chunks, not other files
    if existing:
        for f in existing:
            os.remove(os.path.join(CHUNKS_DIR, f)) # Clean up old chunks

    # Split into chunks (creates chunk_001.csv, chunk_002.csv, ...)
    chunks = [df[i:i+CHUNK_SIZE] for i in range(0, len(df), CHUNK_SIZE)]

    # Save each chunk to a separate CSV file
    # Iterate over each DataFrame slice with its position index (idx=0,1,2... chunk=rows)
    for idx, chunk in enumerate(chunks):
        chunk_file = os.path.join(CHUNKS_DIR, f"chunk_{idx+1:03d}.csv") # Zero-padded chunk number for sorting e.g. chunk_001.csv, chunk_002.csv, ..., chunk_010.csv, chunk_011.csv
        chunk.to_csv(chunk_file, index=False) # Save chunk to CSV without row index

    print(f"\nSummary:")
    print(f"      Total chunks created : {len(chunks)}")
    print(f"      Rows per chunk        : ~{CHUNK_SIZE:,}")
    print(f"      Total rows            : {sum(len(c) for c in chunks):,}")
    print(f"      Output directory      : {os.path.abspath(CHUNKS_DIR)}")

if __name__ == "__main__":
    split_into_chunks()
