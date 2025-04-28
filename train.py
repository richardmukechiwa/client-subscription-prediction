# Initialize Dagshub integration
import dagshub
dagshub.init(repo_owner='richardmukechiwa', repo_name='client-subscription-prediction', mlflow=True)

# Import necessary libraries
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data
test_data = pd.read_csv("artifacts/data_transformation/test.csv")

# Load model and preprocessor
model = joblib.load("artifacts/model_trainer/xgb_model.joblib")  # Make sure you fix the filename
processor = joblib.load("artifacts/model_trainer/xgb_preprocessor.joblib")
encoder = joblib.load("artifacts/model_trainer/xgb_encoder.joblib") 

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

# Set experiment name
mlflow.set_experiment("classification_with_xgb_classifier_artifacts")

# End any existing MLflow run
if mlflow.active_run():
    mlflow.end_run()

# Start a new MLflow run
with mlflow.start_run(run_name="xgb_classifier_best"):

    # Log model with input example and signature
    input_example = test_x.iloc[:5]
    signature = infer_signature(test_x, predictions)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        signature=signature
    )

    # Log model parameters and metrics
    mlflow.log_param("model", "xgb_classifier_best")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log artifacts (e.g., preprocessor, model files)
    mlflow.log_artifacts("artifacts/model_trainer")

    # Set tags for better organization
    mlflow.set_tag("stage", "Evaluation")
    mlflow.set_tag("model_type", "XGBClassifier")
    mlflow.set_tag("dataset", "Client Subscription Prediction")

print(" MLflow run completed successfully and artifacts logged to Dagshub!")
