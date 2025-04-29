# Initialize Dagshub integration
import dagshub
dagshub.init(repo_owner='richardmukechiwa', repo_name='client-subscription-prediction', mlflow=True)

# Import necessary libraries
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Load test data
test_data = pd.read_csv("artifacts/data_transformation/test.csv")

# Load model and preprocessor
model = joblib.load("artifacts/model_trainer/xgb.joblib")
processor = joblib.load("artifacts/model_trainer/xgb_preprocessor.joblib")
encoder = joblib.load("artifacts/model_trainer/label_encoder.joblib") 

# Separate features and target
test_x = test_data.drop(columns="y")
test_y = test_data["y"]

test_x = processor.transform(test_x)
test_y = encoder.transform(test_y)

# Predict on test set
predictions = model.predict(test_x)

# Calculate metrics
accuracy = accuracy_score(test_y, predictions)
precision = precision_score(test_y, predictions)
recall = recall_score(test_y, predictions)
f1 = f1_score(test_y, predictions)

# Get classification report and confusion matrix
clf_report = classification_report(test_y, predictions, output_dict=True)
clf_report_text = classification_report(test_y, predictions)
conf_matrix = confusion_matrix(test_y, predictions)


# Set experiment name
mlflow.set_experiment("classification_with_xgb_classifier_artifacts")

# End any existing MLflow run
if mlflow.active_run():
    mlflow.end_run()

# Start a new MLflow run
with mlflow.start_run(run_name="xgb_classifier_best"):

    # Log model with input example and signature
    input_example = test_x[:5]  # Since it's transformed already
    signature = infer_signature(test_x, predictions)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        signature=signature
    )

    # Log primary metrics
    mlflow.log_param("model", "xgb_classifier_best")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log classification report breakdown (optional but detailed)
    for label, metrics in clf_report.items():
        if isinstance(metrics, dict):
            for metric_name, score in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", score)

    # Log classification report as artifact
    with open("classification_report.txt", "w") as f:
        f.write(clf_report_text)
    mlflow.log_artifact("classification_report.txt")

    # Plot and log confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Log other useful artifacts
    mlflow.log_artifacts("artifacts/model_trainer")

    # Set tags for better organization
    mlflow.set_tag("stage", "Evaluation")
    mlflow.set_tag("model_type", "XGBClassifier")
    mlflow.set_tag("dataset", "Client Subscription Prediction")

print(" MLflow run completed successfully and artifacts logged to Dagshub!")

