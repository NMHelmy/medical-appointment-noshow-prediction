# Nour Helmy - 202202012
# Upload Data to HDFS

import os
from hdfs import InsecureClient
from config import HDFS_URL, HDFS_USER, CHUNKS_DIR, HDFS_INPUT, HDFS_OUTPUT

HDFS_STREAM = "/noshow/streaming_input"

# Connect to real HDFS
client = InsecureClient(HDFS_URL, user=HDFS_USER)

def upload_to_hdfs():
    # Verify chunks exist locally
    chunks = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith('.csv')])
    if not chunks:
        print("No chunks found. Run split_chunks.py first.")
        return

    # Create real HDFS directories
    for d in [HDFS_INPUT, HDFS_STREAM, HDFS_OUTPUT]:
        client.makedirs(d)

    # Upload each chunk to HDFS
    total_size = 0
    for chunk_file in chunks:
        local_path = os.path.join(CHUNKS_DIR, chunk_file)
        hdfs_path  = f"{HDFS_INPUT}/{chunk_file}"
        total_size += os.path.getsize(local_path)

        # Upload file to HDFS (overwrite if exists)
        with open(local_path, 'rb') as f:
            client.write(hdfs_path, f, overwrite=True)

    # Verify by listing HDFS directory
    print(f"\nHDFS directory listing ({HDFS_INPUT}):")
    files = client.list(HDFS_INPUT)
    for f in files:
        print(f"      {f}")

if __name__ == "__main__":
    upload_to_hdfs()
