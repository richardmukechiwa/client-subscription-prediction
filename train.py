import dagshub
dagshub.init(repo_owner='richardmukechiwa', repo_name='client-subscription-prediction', mlflow=True)

import mlflow

# metrics from model
model_name = "xgb_classifier"
accuracy = 0.8209944751381215
precision = 0.8525007003855971
recall = 0.8209944751381215
f1_score = 0.83489495724390

# Set experiment name
mlflow.set_experiment("classification_with xgb_classifier")

# End any existing run
if mlflow.active_run():
    mlflow.end_run()

# Start a new MLflow run
with mlflow.start_run(run_name=model_name):
    mlflow.log_param("model", model_name)  # Logs model name as a parameter
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1_score)
