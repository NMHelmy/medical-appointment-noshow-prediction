# Nour Helmy - 202202012
# Evaluate and compare the three trained models using multiple metrics

import os
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

# Must be set before Spark initializes so its Hadoop client authenticates as the correct user
os.environ["HADOOP_USER_NAME"] = "hadoop"

SPARK_HDFS      = "hdfs://172.20.136.16:9000"
PREDICTIONS_DIR = "/noshow/predictions"

# Ordered by complexity: baseline → ensemble → boosted
MODELS = {
    "Logistic Regression": "lr_predictions",
    "Random Forest":       "rf_predictions",
    "GBT":                 "gbt_predictions"
}

def create_spark_session():
    spark = SparkSession.builder \
        .appName("NoShow_Evaluation") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def compute_metrics(predictions):
    # AUC-ROC: threshold-independent ranking quality; handles class imbalance better than accuracy
    auc_roc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(predictions)

    # AUC-PR: more informative than AUC-ROC when positives (no-show) are the minority class
    auc_pr = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    ).evaluate(predictions)

    mc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # Accuracy is reported but not used for model selection due to the 80/20 class imbalance
    accuracy  = mc.setMetricName("accuracy").evaluate(predictions)
    precision = mc.setMetricName("weightedPrecision").evaluate(predictions)
    recall    = mc.setMetricName("weightedRecall").evaluate(predictions)
    f1        = mc.setMetricName("f1").evaluate(predictions)

    return {
        "AUC-ROC":   auc_roc,
        "AUC-PR":    auc_pr,
        "F1":        f1,
        "Precision": precision,
        "Recall":    recall,
        "Accuracy":  accuracy,
    }

def print_confusion_matrix(predictions, model_name):
    # Collect the 2×2 cell counts from HDFS into a local dict for display
    cm = {
        (int(r["label"]), int(r["prediction"])): r["count"]
        for r in predictions.groupBy("label", "prediction").count().collect()
    }
    print(f"\n  Confusion Matrix — {model_name}:")
    print(f"  {'':>14} Pred=0 (show)   Pred=1 (no-show)")
    for actual in [0, 1]:
        label_str = "Actual=0 (show) " if actual == 0 else "Actual=1 (no-show)"
        c0 = cm.get((actual, 0), 0)
        c1 = cm.get((actual, 1), 0)
        print(f"  {label_str}  {c0:>10,}        {c1:>10,}")

def print_comparison_table(results):
    metrics = ["AUC-ROC", "AUC-PR", "F1", "Precision", "Recall", "Accuracy"]
    col = 12
    sep = "=" * (24 + col * len(metrics))

    print(f"\n{sep}")
    print("MODEL COMPARISON")
    print(sep)
    header = f"{'Model':<24}" + "".join(f"{m:>{col}}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        row = f"{name:<24}" + "".join(f"{scores[m]:>{col}.4f}" for m in metrics)
        print(row)
    print(sep)

def select_best_model(results):
    # Primary criterion: AUC-ROC — threshold-independent and robust under class imbalance
    best  = max(results, key=lambda name: results[name]["AUC-ROC"])
    worst = min(results, key=lambda name: results[name]["AUC-ROC"])

    print(f"\nBest model  : {best}")
    print(f"  AUC-ROC   = {results[best]['AUC-ROC']:.4f}")
    print(f"  AUC-PR    = {results[best]['AUC-PR']:.4f}")
    print(f"  F1        = {results[best]['F1']:.4f}")

    gain = results[best]["AUC-ROC"] - results[worst]["AUC-ROC"]
    print(f"\n{best} outperforms {worst} by {gain:.4f} AUC-ROC points.")
    print("Justification: highest AUC-ROC means it best ranks no-show patients above "
          "show-up patients across all decision thresholds, making it the most reliable "
          "model for downstream triage or reminder systems.")

def main():
    spark = create_spark_session()
    results = {}

    for model_name, folder in MODELS.items():
        path = f"{SPARK_HDFS}{PREDICTIONS_DIR}/{folder}"
        print(f"\n{'─' * 50}")
        print(f"Evaluating: {model_name}")
        print(f"{'─' * 50}")

        predictions = spark.read.parquet(path)
        # Cache each prediction set since multiple evaluators will scan it
        predictions.cache()

        results[model_name] = compute_metrics(predictions)

        scores = results[model_name]
        print(f"  AUC-ROC  : {scores['AUC-ROC']:.4f}")
        print(f"  AUC-PR   : {scores['AUC-PR']:.4f}")
        print(f"  F1       : {scores['F1']:.4f}")
        print(f"  Precision: {scores['Precision']:.4f}")
        print(f"  Recall   : {scores['Recall']:.4f}")
        print(f"  Accuracy : {scores['Accuracy']:.4f}")

        print_confusion_matrix(predictions, model_name)
        predictions.unpersist()

    print_comparison_table(results)
    select_best_model(results)

    spark.stop()

if __name__ == "__main__":
    main()
