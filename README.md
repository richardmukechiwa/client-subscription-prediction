# 💼 Bank Client Subscription Prediction

A machine learning project to predict whether a client will subscribe to a term deposit using banking campaign data. The model is built with XGBoost, optimized using Optuna, and explained using SHAP values. It is deployed using Docker and Render.

## Project Overview

This project is a machine learning classification model designed to predict whether a customer will subscribe to a term deposit based on demographic and campaign-related features. The data comes from a marketing campaign conducted by a Portuguese bank, where customers were contacted via phone calls.

A term deposit is a financial product where a customer deposits a fixed amount of money into a bank account for a specified period and earns interest over that term. It offers:

A guaranteed interest rate

A fixed term duration

Low risk, but with limited early withdrawal options

Users interact with the model via a simple web interface built with Flask. Upon submitting client data (such as age, balance, day of month. month, and outcome of previous campaigns), the model returns a real-time prediction indicating the likelihood of a successful subscription.

 🔗 **Live App:** [https://bank-client-marketing-predictor.onrender.com](https://bank-client-marketing-predictor.onrender.com)

### Dataset Source

Name: Bank Marketing Dataset

Source: UCI Machine Learning Repository

Description: Contains data related to direct marketing campaigns of a Portuguese banking institution, including both categorical and numerical features.

### Tech Stack

- Python for backend logic and model development

- Pandas & scikit-learn for data preprocessing and modeling

- Flask for web application development

- HTML/CSS for frontend user interface

- Jupyter Notebook for exploratory data analysis and prototyping

- Joblib for saving the trained model and pipeline


## ⚙️ Technologies Used

- **Python**
- **Logistic Regression**
- **Random Forest**
- **XGBoost** with **Optuna** for hyperparameter tuning
- **SMOTE** for class balancing
- **SHAP** for model explainability
- **Docker** for containerization
- **Render** for deployment
- **Flask** for interactive UI

## 🔍 Key Results

- ✅ **Best Model**: XGBoost with Optuna tuning
- 🎯 **F1 Score**: 1.00 at optimized threshold (0.48)
- 💡 **Top Features (via SHAP)**:
  - `poutcome_success`
  - `month_5`
  - `balance`
  - `age`
  - `day`

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the test
10. Update the app.py

# How to run the project

Clone the repository below:
https://github.com/richardmukechiwa/client-subscription-prediction
...
### STEP 01 - Create a conda environment after opening the repository



```python
conda create -n cl python=3.10 -y
```

```python
conda activate cl
```

...
### STEP 02 - install the requirements

```python
pip install -r requirements.txt
```

```python
# Run python main.py to train the model
python main.py 
```

```python
#Run python app.py to get the predictions
python app.py 
```

### Mlflow

```python
# install mlflow
pip install mlflow
```

```python
#run the code below to start MLFLOW
mlflow ui
```

```python
# example code for logging metrics
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    model = XGBClassifier()
    model.fit(X_train, y_train)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))
    mlflow.sklearn.log_model(model, "model")

```

 **Build Docker Image**
   ```bash
   docker build -t bank-prediction-app .
   ```

3. **Run the App**
   ```bash
   docker run -p 8501:8501 bank-prediction-app
   ```

4. **Visit** `http://localhost:8501` in your browser.

---

## 📈 Use Case

This app helps banks:
- Focus marketing efforts on high-potential clients
- Save on marketing costs
- Improve campaign success rates

---

## 🙋 About Me

👋 I'm Richard Mukechiwa, a data science enthusiast with a passion for solving real-world problems with machine learning.

🔗 [LinkedIn](https://www.linkedin.com/in/richardmukechiwa)  
📬 [mukechiwarichard@gmail.com](mailto:mukechiwarichard@gmail.com)

---

## 🏷️ Tags

`#MachineLearning` `#XGBoost` `#SMOTE` `#SHAP` `#Docker` `#Streamlit` `#Render` `#Banking` `#Deployment` `#OpenToWork`

---

## ⭐ Acknowledgements

Inspired by the UCI Bank Marketing dataset and XGBoost documentation.

If you like this project, feel free to ⭐ the repo and share it!





