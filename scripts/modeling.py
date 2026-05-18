# Nour Helmy - 202202012
# Train three classifiers on the processed no-show dataset using PySpark MLlib
# Logistic Regression, Random Forest, and Gradient Boosted Trees (GBT)

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier, GBTClassifier)

# Must be set before Spark initializes so its Hadoop client authenticates as the correct user
os.environ["HADOOP_USER_NAME"] = "hadoop"

SPARK_HDFS      = "hdfs://172.20.136.16:9000"
PROCESSED_PATH  = "/noshow/output/final_processed"
MODELS_DIR      = "/noshow/models"
PREDICTIONS_DIR = "/noshow/predictions"

# AppointmentID is an identifier, checksum is an MD5 hash — both are excluded
FEATURE_COLS = [
    "Age", "Scholarship", "Hipertension", "Diabetes",
    "Alcoholism", "Handcap", "SMS_received",
    "WaitDays", "GenderEncoded", "AppointmentWeekday"
]

def create_spark_session():
    spark = SparkSession.builder \
        .appName("NoShow_Modeling") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_data(spark):
    path = f"{SPARK_HDFS}{PROCESSED_PATH}"
    df = spark.read.parquet(path)
    print(f"Loaded {df.count():,} rows from {path}")
    return df

def add_class_weights(df):
    # Dataset is ~80% label=0 (showed up) and ~20% label=1 (no-show)
    # Class weights penalize misclassifying the minority class more heavily during training
    total  = df.count()
    counts = {r["label"]: r["count"] for r in df.groupBy("label").count().collect()}    
    w0 = total / (2 * counts[0])
    w1 = total / (2 * counts[1])
    print(f"Class counts — label=0: {counts[0]:,}  label=1: {counts[1]:,}")
    print(f"Class weights — w0: {w0:.4f}  w1: {w1:.4f}")
    return df.withColumn("classWeight", F.when(F.col("label") == 1, w1).otherwise(w0))

def assemble_features(df):
    # VectorAssembler merges individual feature columns into a single dense vector for MLlib
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    return assembler.transform(df).drop("AppointmentID", "checksum")

def scale_features(train_df, test_df):
    # Logistic Regression is sensitive to the size of numbers - Random Forest and GBT dont need this
    # (e.g. Age ranges from 0 to 100, while Scholarship is binary 0/1), so we standardize them
    # Scaling normalizes everything to the same range so no column dominates unfairly.
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=False, withStd=True)
    scaler_model = scaler.fit(train_df)
    return scaler_model.transform(train_df), scaler_model.transform(test_df)

def print_feature_importances(model, label):
    importances = sorted(zip(FEATURE_COLS, model.featureImportances.toArray()), key=lambda x: x[1], reverse=True)
    print(f"  Feature importances ({label}):")
    for name, score in importances:
        print(f"    {name:<22} {score:.4f}")

def save_predictions(predictions, folder_name):
    path = f"{SPARK_HDFS}{PREDICTIONS_DIR}/{folder_name}"
    # rawPrediction is required by BinaryClassificationEvaluator for AUC calculations
    predictions.select("label", "prediction", "probability", "rawPrediction") \
        .coalesce(1).write.mode("overwrite").parquet(path)
    print(f"  Predictions saved → {path}")

def train_logistic_regression(train_scaled, test_scaled):
    # The model starts with a random guess, then repeats this loop up to 100 times:
    # 1) Make predictions on the training data using current coefficients
    # 2) Compare predictions to actual labels to compute the error
    # 3) Adjust coefficients slightly to reduce the error (gradient descent)
    print("\n[1/3] Logistic Regression")
    lr = LogisticRegression(
        featuresCol="scaledFeatures",
        labelCol="label",
        weightCol="classWeight",
        maxIter=100,
        regParam=0.01,       # L2 regularization to prevent overfitting
        elasticNetParam=0.0  # pure L2 (Ridge); set >0 to mix in L1 (Lasso)
    )
    model = lr.fit(train_scaled)
    preds = model.transform(test_scaled)
    model.write().overwrite().save(f"{SPARK_HDFS}{MODELS_DIR}/logistic_regression")
    save_predictions(preds, "lr_predictions")
    print(f"  Intercept: {model.intercept:.4f}")
    return model


def train_random_forest(train_df, test_df):
    # It builds 100 decision trees independently, each one trained on a slightly different random sample of the data
    print("\n[2/3] Random Forest")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        numTrees=100,   # more trees reduce variance but increase training time
        maxDepth=10,
        seed=42
    )
    model = rf.fit(train_df)
    preds = model.transform(test_df)
    model.write().overwrite().save(f"{SPARK_HDFS}{MODELS_DIR}/random_forest")
    save_predictions(preds, "rf_predictions")
    print_feature_importances(model, "Random Forest")
    return model

def train_gbt(train_df, test_df):
    # Unlike RF which builds trees independently, GBT builds them one after another, each fixing the previous one's mistakes
    print("\n[3/3] Gradient Boosted Trees")
    # GBTClassifier does not support weightCol in PySpark MLlib
    # The 80/20 imbalance is moderate enough that GBT still learns a useful boundary
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=50,    # number of boosting rounds; more rounds risk overfitting
        maxDepth=5,    # shallower trees than RF to prevent individual tree overfitting
        seed=42
    )
    model = gbt.fit(train_df)
    preds = model.transform(test_df)
    model.write().overwrite().save(f"{SPARK_HDFS}{MODELS_DIR}/gbt")
    save_predictions(preds, "gbt_predictions")
    print_feature_importances(model, "GBT")
    return model

def main():
    spark = create_spark_session()

    df = load_data(spark)
    df = add_class_weights(df)
    df = assemble_features(df)

    # Split before scaling so the scaler never sees test data statistics
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"\nTrain: {train_df.count():,} rows  |  Test: {test_df.count():,} rows")

    train_scaled, test_scaled = scale_features(train_df, test_df)

    # Cache all four DataFrames since each model reads them fully during training and prediction
    train_df.cache()
    test_df.cache()
    train_scaled.cache()
    test_scaled.cache()

    train_logistic_regression(train_scaled, test_scaled)
    train_random_forest(train_df, test_df)
    train_gbt(train_df, test_df)

    train_df.unpersist()
    test_df.unpersist()
    train_scaled.unpersist()
    test_scaled.unpersist()

    spark.stop()
    print("\nModeling complete. Run evaluation.py to compare results.")


if __name__ == "__main__":
    main()
