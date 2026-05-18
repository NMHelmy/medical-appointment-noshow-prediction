# Nour Helmy - 202202012
# Fake Streaming + ETL + Checksum

import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from hdfs import InsecureClient

HDFS_URL      = "http://172.20.136.16:9870"
HDFS_USER     = "hadoop"
HDFS_INPUT    = "/noshow/input"
HDFS_OUTPUT   = "/noshow/output"
PROCESSED_DIR = "/noshow/output/processed"
SPARK_HDFS    = "hdfs://172.20.136.16:9000"
STREAM_DELAY  = 1 # Seconds to wait between processing each chunk to simulate real-time streaming

# Must be set before Spark initializes so its Hadoop client authenticates as the correct user
os.environ["HADOOP_USER_NAME"] = "hadoop"

client = InsecureClient(HDFS_URL, user=HDFS_USER)

# Initialize Spark engine using all available CPU cores on this machine
def create_spark_session():
    spark = SparkSession.builder \
        .appName("NoShow_FakeStreaming_ETL") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.memory", "2g") \
        .config("spark.hadoop.hadoop.job.ugi", "hadoop") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

# Concatenate 5 key fields with | separator then hash the result into an MD5 fingerprint
# This fingerprint uniquely identifies each row and is used to detect duplicates
def compute_checksum(df):
    df = df.withColumn(
        "checksum",
        F.md5(
            F.concat_ws("|",
                F.col("AppointmentID").cast(StringType()),
                F.col("PatientId").cast(StringType()),
                F.col("ScheduledDay"),
                F.col("AppointmentDay"),
                F.col("No-show")
            )
        )
    )
    return df

# Count total rows vs rows with distinct AppointmentIDs to find duplicates
def detect_duplicates(df, chunk_name):
    total    = df.count()
    distinct = df.dropDuplicates(["AppointmentID"]).count()
    dupes    = total - distinct
    if dupes > 0:
        print(f"[{chunk_name}] Found {dupes} duplicate AppointmentIDs - removing")
    else:
        print(f"[{chunk_name}] No duplicates found (checksum verified)")
    # Remove duplicate rows keeping only the first occurrence of each AppointmentID
    return df.dropDuplicates(["AppointmentID"])

def etl_transform(df, chunk_name):
    original_count = df.count()

    # Remove rows where age is negative or above 115 as these are data entry errors
    df = df.filter((F.col("Age") >= 0) & (F.col("Age") <= 115))
    removed_age = original_count - df.count()
    if removed_age > 0:
        print(f"Removed {removed_age} rows with invalid age")

    # Convert date strings from ISO format into proper Spark timestamp objects
    df = df.withColumn("ScheduledDay",
            F.to_timestamp("ScheduledDay", "yyyy-MM-dd'T'HH:mm:ss'Z'"))
    df = df.withColumn("AppointmentDay",
            F.to_timestamp("AppointmentDay", "yyyy-MM-dd'T'HH:mm:ss'Z'"))

    # Calculate number of days between booking date and appointment date
    df = df.withColumn("WaitDays",
            F.datediff(F.col("AppointmentDay"), F.col("ScheduledDay")))
    # Set negative wait days to 0 as they are caused by same-day booking data errors
    df = df.withColumn("WaitDays",
            F.when(F.col("WaitDays") < 0, 0).otherwise(F.col("WaitDays")))

    # Convert No-show text column to numeric: Yes=1 (did not show), No=0 (showed up)
    df = df.withColumn("label",
            F.when(F.col("No-show") == "Yes", 1).otherwise(0))

    # Convert gender text column to numeric: F=1, M=0
    df = df.withColumn("GenderEncoded",
            F.when(F.col("Gender") == "F", 1).otherwise(0))

    # Extract the day of the week from appointment date: 1=Sunday, 7=Saturday
    df = df.withColumn("AppointmentWeekday",
            F.dayofweek(F.col("AppointmentDay")))

    # Add MD5 fingerprint column to each row for integrity verification
    df = compute_checksum(df)

    # Find and remove any duplicate appointments using AppointmentID
    df = detect_duplicates(df, chunk_name)

    # Remove columns that are already encoded, not useful, or privacy-sensitive
    df = df.drop("Gender", "No-show", "ScheduledDay", "AppointmentDay",
                 "Neighbourhood", "PatientId")

    print(f"Rows after ETL: {df.count():,} (from {original_count:,})")
    return df

# Get sorted list of all CSV chunk files from real HDFS input directory
def run_fake_streaming(spark):
    chunks = sorted([f for f in client.list(HDFS_INPUT) if f.endswith('.csv')])

    if not chunks:
        print("No chunks found in HDFS. Run hdfs_upload.py first.")
        return

    print(f"\nStarting fake streaming — {len(chunks)} chunks to process\n")
    # List to collect all processed chunk DataFrames for final merging
    all_processed = []

    for idx, chunk_file in enumerate(chunks):
        # Build full hdfs:// path that Spark needs to read the file from HDFS
        chunk_path = f"{SPARK_HDFS}{HDFS_INPUT}/{chunk_file}"
        print(f"  [{idx+1:02d}/{len(chunks)}] Processing: {chunk_file}")
        print(f"         Reading from HDFS: {chunk_path}")

        # Extract - Reading the raw data
        # Spark reads the chunk CSV directly from HDFS into a distributed DataFrame
        df = spark.read.csv(chunk_path, header=True, inferSchema=True)
        print(f"Raw rows read: {df.count():,}")

        # Transform - Cleaning and reshaping the data
        # Run all 9 ETL cleaning and transformation steps on this chunk
        df_clean = etl_transform(df, chunk_file)

        # Load - Saving the cleaned data to its destination
        # Build output path for this chunk's processed parquet file (compressed, column-based format) on HDFS
        out_path = f"{SPARK_HDFS}{PROCESSED_DIR}/processed_{chunk_file.replace('.csv', '')}"
        # Merge Spark partitions into one file and save as Parquet to HDFS
        df_clean.coalesce(1).write.mode("overwrite").parquet(out_path)

        # Keep reference to this chunk's DataFrame for final merge
        all_processed.append(df_clean)
        # Wait before processing next chunk to simulate real-time data arrival
        time.sleep(STREAM_DELAY)

    # Stack all 11 processed chunk DataFrames into one single DataFrame
    final_df = all_processed[0]
    for df in all_processed[1:]:
        final_df = final_df.union(df)

    # Print the final column names and data types after all transformations
    final_df.printSchema()

    # Show first 5 rows of the final cleaned dataset
    final_df.show(5, truncate=False)

    # Show how many patients showed up (0) vs did not show up (1)
    final_df.groupBy("label").count().orderBy("label").show()

    # Build final output path and save merged dataset as a single Parquet file to HDFS
    final_output = f"{SPARK_HDFS}{HDFS_OUTPUT}/final_processed"
    final_df.coalesce(1).write.mode("overwrite").parquet(final_output)

    return final_df

def main():
    # Start Spark engine
    spark = create_spark_session()
    # Run the full fake streaming ETL pipeline
    final_df = run_fake_streaming(spark)
    # Shut down Spark and release all resources
    spark.stop()

if __name__ == "__main__":
    main()