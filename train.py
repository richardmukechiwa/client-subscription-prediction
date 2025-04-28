import dagshub
dagshub.init(repo_owner='richardmukechiwa', repo_name='client-subscription-prediction', mlflow=True)

import mlflow

# metrics from model
model_name = "xgb_classifier_best"  
accuracy = 0.8552486187845304
precision = 0.86349882593394
recall = 0.8552486187845304
f1_score =  0.859162770816417


# Set experiment name
mlflow.set_experiment("classification_with xgb_classifier_artifacts")

# End any existing run
if mlflow.active_run():
    mlflow.end_run()

# Start a new MLflow run
with mlflow.start_run(run_name=model_name):
    mlflow.log_artifacts("")
    mlflow.log_param("model", model_name)  # Logs model name as a parameter
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1_score)
