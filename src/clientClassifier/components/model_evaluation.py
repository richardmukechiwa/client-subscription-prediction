import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from urllib.parse import urlparse
import joblib
import mlflow 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import shap
from clientClassifier.utils.common import save_json
from clientClassifier.entity.config_entity import ModelEvaluationConfig
from clientClassifier    import logger
import numpy as np

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
       
    def feature_importance(self):
        
        test_data = pd.read_csv(self.config.test_data_path)
        
        test_x = test_data.drop(self.config.target_column, axis=1)
        
        pipeline = joblib.load(self.config.xgb_pipeline_eval)
        
        # change the month column to string
        test_x['month'] = test_x['month'].astype('str')
        
         # Extract preprocessor and model from pipeline
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['classifier']

        # Transform test data using preprocessor
        X_processed = preprocessor.transform(test_x)

        # Get feature names after preprocessing
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            num_features = preprocessor.transformers_[0][2]
            cat_encoder = preprocessor.transformers_[1][1]
            cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
            feature_names = np.concatenate([num_features, cat_features])

        # Convert processed features to DataFrame
        X_df = pd.DataFrame(
            X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed,
            columns=feature_names
        )

        # Create SHAP explainer and compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)

        # Generate SHAP plot
        shap.summary_plot(shap_values, X_df, show=False)
        #buf = BytesIO()
        #plt.savefig(buf, format="png", bbox_inches='tight')
        #plt.close()
        #buf.seek(0)

        # Encode image to base64
        #img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        #return img_base64
        
        
        
        #findings of the shap summary plot
        
        logger.info("The SHAP analysis revealed that the top five features driving model predictions are:")
        print("  " * 50)
        logger.info("1. Previous Campaign Outcome (cat__poutcome_success): Clients who previously responded positively to campaigns are much more likely to subscribe again.")
        print(" " * 50)
        logger.info("2. Account Balance (num__balance): Clients with higher account balances are significantly more likely to subscribe.") 
        print(" " * 50)
        logger.info("3. Day of Contact (num__day): The specific day of the month when a client is contacted strongly influences the outcome.")
        print(" " * 50)
        logger.info("4. Month of Contact (num__month): Seasonality effects are evident, with specific months (e.g., holiday or bonus periods) boosting subscription likelihood.")
        print(" " * 50)
        logger.info("5. Age of Client (num__age): Age plays a significant role, with different age groups showing distinct patterns in subscription behavior.")                 

    
        
       
    #Evaluation and mlflow  on selected features
    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        return accuracy, precision, recall, f1

    def log_confusion_matrix(self, actual, predicted, class_names):
        cm = confusion_matrix(actual, predicted)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        temp_img_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        plt.savefig(temp_img_path)
        plt.close()

        mlflow.log_artifact(temp_img_path, artifact_path="confusion_matrix")

    def log_classification_report(self, actual, predicted, class_names):
        report = classification_report(actual, predicted, target_names=class_names)
        temp_txt_path = tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name
        with open(temp_txt_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(temp_txt_path, artifact_path="xgb_classification_report")

    def log_selected_features_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        
        model = joblib.load(self.config.xgb_pipeline_eval)
        encoder = joblib.load(self.config.xgb_encoder)
    

        test_x = test_data.drop(self.config.target_column, axis=1)
        test_y = test_data[self.config.target_column]
        
        test_x['month'] = test_x['month'].astype('str')
        
        #introduce selected features
        important_features = ['age', 'month', 'day', 'balance', 'poutcome']
        
        test_x  = test_x[important_features]
        
        #label encoding the target variable
        test_y = encoder.transform(test_y)
        

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        mlflow.set_experiment("classification_with_xgbclassifier_on_selected_features")
        

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run():
            #processing = model.transform(test_x)
           
            predicted_qualities = model.predict(test_x)
            
            
            accuracy, precision, recall, f1 = self.eval_metrics(test_y, predicted_qualities)
            
            # evaluate the model
            xgb_report = classification_report(test_y, predicted_qualities)
            xgb_cm = confusion_matrix(test_y, predicted_qualities)   
            xgb_accuracy = accuracy_score(test_y, predicted_qualities)   
            
            #create Confusion Matrix Display
            cm_display = ConfusionMatrixDisplay(confusion_matrix=xgb_cm, display_labels=encoder.classes_)
            cm_display.plot()
            plt.title("XGBClassifier Matrix")

            logger.info(f"XGBoost Classification Report:\n{xgb_report}")
            logger.info(f"XGBoost Confusion Matrix:\n{xgb_cm}") 
            logger.info(f"XGBoost Accuracy: {xgb_accuracy}")
            
            
           

            model_name = "xgb_classifier_best" 

            scores = {
                "model_name": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            class_names =encoder.classes_
            self.log_confusion_matrix(test_y, predicted_qualities, class_names)
            self.log_classification_report(test_y, predicted_qualities, class_names)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="XGB ClassificationModel")
            else:
                mlflow.sklearn.log_model(model, "model")
                
   
            return xgb_report, xgb_cm, xgb_accuracy