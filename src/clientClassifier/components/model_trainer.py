import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from clientClassifier import logger
from clientClassifier.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def train_XGBClassifier(self):
        logger.info("Training XGBoost model")

        # Load data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Separate features and target
        X_train = train_data.drop(columns=[self.config.target_column])
        y_train = train_data[self.config.target_column]

        X_test = test_data.drop(columns=[self.config.target_column])
        y_test = test_data[self.config.target_column]
        
        
        # Apply Label Encoding to target variable
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        


        # calculating "scale_pos_weight"
        neg = np.sum(y_train == 0)  # count of negative class
        pos = np.sum(y_train == 1)  # count of positive class

        scale_pos_weight = neg / pos
        print(f"scale_pos_weight = {scale_pos_weight:.2f}")
        
        


        # Define numerical and categorical features
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

        # Transformers
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        xgb_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Preprocess X_train using fit_transform (fit only on training data)
        X_train_processed = xgb_preprocessor.fit_transform(X_train)

        # Fitting XGBoost model on the processed data
        xgb_model = XGBClassifier(random_state=self.config.random_state,
                                    n_estimators=self.config.n_estimators,
                                    max_depth=self.config.max_depth,
                                    scale_pos_weight=scale_pos_weight)
        
            
        xgb_model.fit(X_train_processed, y_train)

        # make predictions on the test data 
        X_test_processed = xgb_preprocessor.transform(X_test)
        y_pred_xgb = xgb_model.predict(X_test_processed)

        # evaluate the model
        xgb_report = classification_report(y_test, y_pred_xgb)
        xgb_cm = confusion_matrix(y_test, y_pred_xgb)   
        xgb_accuracy = accuracy_score(y_test, y_pred_xgb)   
        
        #create Confusion Matrix Display
        cm_display = ConfusionMatrixDisplay(confusion_matrix=xgb_cm, display_labels=label_encoder.classes_)
        cm_display.plot()
        plt.title("XGBClassifier Matrix")

        logger.info(f"XGBoost Classification Report:\n{xgb_report}")
        logger.info(f"XGBoost Confusion Matrix:\n{xgb_cm}") 
        logger.info(f"XGBoost Accuracy: {xgb_accuracy}")
        
        # save the model
        xgb_label_path = os.path.join(self.config.root_dir, self.config.label_encoder_names )
        xgb_model_path = os.path.join(self.config.root_dir, self.config.xgb_model_name)   
        xgb_preprocessor_path = os.path.join(self.config.root_dir, self.config.xgb_preprocessor_name)
        
        joblib.dump(label_encoder, xgb_label_path)
        joblib.dump(xgb_model, xgb_model_path)
        joblib.dump(xgb_preprocessor, xgb_preprocessor_path)
        return xgb_report, xgb_cm, xgb_accuracy