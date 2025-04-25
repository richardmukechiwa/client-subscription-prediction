import dagshub
dagshub.init(repo_owner='richardmukechiwa', repo_name='client-subscription-prediction', mlflow=True)

import mlflow

# Your metrics from model
model_name = "random_forest"
accuracy = 0.7535911602209945
precision = 0.8500484417714119
recall = 0.7535911602209945
f1 = 0.7909325678172326

# Set experiment name
mlflow.set_experiment("classification_with random forest")

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
