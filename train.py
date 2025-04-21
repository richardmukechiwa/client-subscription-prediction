import dagshub
from dagshub import dagshub_logger
import mlflow
import os

dagshub.init(repo_owner='richardmukechiwa', repo_name='client-subscription-prediction', mlflow=True)

# Metrics
model_name = "LogisticRegression"
accuracy = 0.6685082872928176
precision = 0.8357025086402515
recall = 0.6685082872928176
f1_score = 0.7287747717155298

# Paths to artifacts (update these as needed)
model_path = "artifacts/model_trainer/sm_model_pipeline.joblib"
preprocessor_path = "artifacts/model_trainer/sm_preprocessor.joblib"
metrics_json_path = "artifacts/model_evaluation/metrics.json"

# Set experiment name
mlflow.set_experiment("classification_with_SMOTE")

if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name=model_name):
    # Log params and metrics
    mlflow.log_param("model", model_name)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1_score)

    # Log individual artifacts
    mlflow.log_artifact(model_path, artifact_path="models")
    mlflow.log_artifact(preprocessor_path, artifact_path="preprocessors")
    mlflow.log_artifact(metrics_json_path, artifact_path="metrics")

    # OPTIONAL: log entire folder (e.g., all models)
    # mlflow.log_artifacts("artifacts/model_trainer", artifact_path="model_trainer_all")

