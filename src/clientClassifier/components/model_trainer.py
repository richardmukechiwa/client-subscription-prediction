import os

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

import optuna

from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from clientClassifier import logger
from clientClassifier.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train_selected_features(self):
        logger.info("Training XGBClassifier model on selected features")

        # Load data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Separate features and target
        X_train = train_data.drop(columns=[self.config.target_column])
        y_train = train_data[self.config.target_column]
        
        
        
        important_features = ['age', 'month', 'day', 'balance', 'poutcome']
        X_train = X_train[important_features]

        #X_test = test_data.drop(columns=[self.config.target_column])
        y_test = test_data[self.config.target_column]
        
        # Label Encoding
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        
        # Scale Pos Weight
        neg = np.sum(y_train == 0)
        pos = np.sum(y_train == 1)
        scale_pos_weight = neg / pos
        print(f"scale_pos_weight = {scale_pos_weight:.2f}")
        
        
        # Convert month to string
        X_train["month"] = X_train["month"].astype(str)
        
    
        # Preprocessing
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical features: {numerical_features}")
        print(f"Categorical features: {categorical_features}")

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        xgb_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Preprocess
        X_train_processed = xgb_preprocessor.fit_transform(X_train)
        
        #print(X_train_processed[20:30])

        #  Define Optuna objective function
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'scale_pos_weight': scale_pos_weight,
                'random_state': self.config.random_state,
                'eval_metric': 'auc' 
            }

            
            model = XGBClassifier(**params)
            model.fit(X_train_processed, y_train)

            preds = model.predict(X_train_processed)  
            acc = accuracy_score(y_train, preds)
            
            return acc  # maximized accuracy

        # Create Optuna study
        study = optuna.create_study(direction="maximize", study_name="xgb_selected_features_optimization")
        study.optimize(objective, n_trials=30)  

        print("Best parameters:", study.best_params)

        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.config.random_state,
           
            'eval_metric': 'auc'
        })

        # Define full pipeline with preprocessor and classifier
        final_pipeline = Pipeline([
            ('preprocessor', xgb_preprocessor),
            ('classifier', XGBClassifier(**best_params))
        ])

        # Fit the pipeline on raw X_train
        final_pipeline.fit(X_train, y_train)
        
        # 1. Get predicted probabilities for the positive class
        y_probs = final_pipeline.predict_proba(X_train)[:, 1]

        # 2. Find the best threshold based on F1-score
        best_threshold = 0.5
        best_f1 = 0

        for thresh in np.arange(0.1, 0.9, 0.01):
            y_pred_thresh = (y_probs >= thresh).astype(int)
            f1 = f1_score(y_train, y_pred_thresh)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        print(f"Best threshold: {best_threshold:.2f} with F1 score: {best_f1:.4f}")
                
        # Save everything
        xgb_label_path = os.path.join(self.config.root_dir, self.config.label_encoder_names )
        xgb_model_path = os.path.join(self.config.root_dir, self.config.xgb_pipeline)
        xgb_processor_path= os.path.join(self.config.root_dir, self.config.xgb_preprocessor_name)  
  

        joblib.dump(label_encoder, xgb_label_path)
        joblib.dump(final_pipeline, xgb_model_path)
        joblib.dump(xgb_preprocessor, xgb_processor_path)
    

        logger.info("Training completed and artifacts saved!")