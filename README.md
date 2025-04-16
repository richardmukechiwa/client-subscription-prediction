# Client-Subscription-prediction

This project aims to predict whether a loan application will be approved based on various client information. Utilizing machine learning techniques, the model analyzes key factors such as age, loan amount, income, and other relevant details to make accurate predictions.

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the tests
10. Update the dvc.py
11. Update the app.py


## Dagshub

[[dagshub](https://dagshub.com)

Run this the code below to track the project on Dagshub:

#####  Use Your Own Parameters

```python

import dagshub
dagshub.init(repo_owner='richardmukechiwa', repo_name='client-subscription-prediction', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

```


