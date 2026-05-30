# Nour Helmy - 202202012
# Shared configuration for all pipeline scripts

import os

HDFS_HOST     = os.environ.get("HDFS_HOST", "172.20.136.16")
HDFS_USER     = "hadoop"
HDFS_URL      = f"http://{HDFS_HOST}:9870"
SPARK_HDFS    = f"hdfs://{HDFS_HOST}:9000"

DATA_DIR      = os.environ.get("DATA_DIR", "../data")
RAW_FILE      = f"{DATA_DIR}/raw/appointments.csv"
CHUNKS_DIR    = f"{DATA_DIR}/chunks"

HDFS_INPUT    = "/noshow/input"
HDFS_OUTPUT   = "/noshow/output"
PROCESSED_DIR = "/noshow/output/processed"
HASH_STORE    = "/noshow/output/hash_store"
PREDICTIONS_DIR = "/noshow/predictions"
MODELS_DIR    = "/noshow/models"
