#!/bin/bash
set -e

echo "=== Step 1/5: Splitting dataset into chunks ==="
python scripts/split_chunks.py

echo "=== Step 2/5: Uploading chunks to HDFS ==="
python scripts/hdfs_upload.py

echo "=== Step 3/5: Fake streaming ETL ==="
python scripts/fake_streaming_etl.py

echo "=== Step 4/5: Training models ==="
python scripts/modeling.py

echo "=== Step 5/5: Evaluating models ==="
python scripts/evaluation.py

echo "=== Pipeline complete ==="
