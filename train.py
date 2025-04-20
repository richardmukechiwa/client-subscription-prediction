import dagshub
dagshub.init(repo_owner='richardmukechiwa', repo_name='client-subscription-prediction', mlflow=True)

import mlflow

# Your metrics from model
model_name = "LogisticRegression"
accuracy = 0.6762430939226519
precision = 0.8398989283395172
recall = 0.6762430939226519
f1 = 0.7348829408102018

# Set experiment name
mlflow.set_experiment("classification_baseline")

# End any existing run
if mlflow.active_run():
    mlflow.end_run()

# Start a new MLflow run
with mlflow.start_run(run_name=model_name):
    mlflow.log_param("model", model_name)  # Logs model name as a parameter
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
