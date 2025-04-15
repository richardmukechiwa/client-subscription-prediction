import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from clientClassifier import logger
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE    
from clientClassifier.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    def train(self):
        logger.info("Training model")
        
        # load train and test data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        # separate features and target variable
        X_train = train_data.drop(columns=[self.config.target_column], axis=1)
        y_train = train_data[self.config.target_column]
        
        X_test = test_data.drop(columns=[self.config.target_column], axis=1)
        y_test = test_data[self.config.target_column]
        
        # create preprocessing pipeline for numerical and categorical features
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()       
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()   

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')                
        
        preprocessor = ColumnTransformer(   
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )   
        
        # create a pipeline with preprocessing and model
        pipeline = Pipeline([   
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                class_weight=self.config.class_weight,
                max_iter=self.config.max_iter,
                penalty=self.config.penalty,
                C=self.config.C,
                solver = self.config.solver,
                random_state=self.config.random_state
            ))
        ])      
        
        # fit the pipeline on the training data
        pipeline.fit(X_train, y_train)      
        
     
        # make predictions on the test data 
        y_pred = pipeline.predict(X_test)      
         
        # evaluate the model
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)   
        accuracy = accuracy_score(y_test, y_pred)   

        logger.info(f"Classification Report:\n{report}")
        logger.info(f"Confusion Matrix:\n{cm}") 
        logger.info(f"Accuracy: {accuracy}")   
         
        # save the model    
        
        
        model = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor'] 
        model_path = os.path.join(self.config.root_dir, self.config.model_name)                     
        preprocessor_path = os.path.join(self.config.root_dir, self.config.preprocessor_name)
        joblib.dump(preprocessor, preprocessor_path)    
        joblib.dump(model, model_path)  
        
        logger.info(f"Model saved at: {model_path}")
        logger.info(f"Preprocessor saved at: {preprocessor_path}")      
        
        # save the model
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
        
  
    def train_with_SMOTE(self):
        logger.info("Training model with SMOTE")

        # Load data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Separate features and target
        X_train = train_data.drop(columns=[self.config.target_column])
        y_train = train_data[self.config.target_column]

        X_test = test_data.drop(columns=[self.config.target_column])
        y_test = test_data[self.config.target_column]

        # Define numerical and categorical features
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

        # Transformers
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        sm_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        #  Preprocess X_train using fit_transform (fit only on training data)
        X_train_processed = sm_preprocessor.fit_transform(X_train)

        # Applying SMOTE to resample training data
        smote = SMOTE(random_state=self.config.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)

        logger.info(f"After SMOTE: {X_resampled.shape}, Target distribution: {dict(pd.Series(y_resampled).value_counts())}")

        #  Fitting model on resampled data
        model = LogisticRegression(
            class_weight=self.config.class_weight,
            max_iter=self.config.max_iter,
            penalty=self.config.penalty,
            C=self.config.C,
            solver = self.config.solver,
            random_state=self.config.random_state
        )

        model.fit(X_resampled, y_resampled)
            
        # make predictions on the test data 
        X_test_processed = sm_preprocessor.transform(X_test)
        y_pred_resampled = model.predict(X_test_processed)

        # evaluate the model
        resampled_report = classification_report(y_test, y_pred_resampled)
        resampled_cm = confusion_matrix(y_test, y_pred_resampled)   
        resampled_accuracy = accuracy_score(y_test, y_pred_resampled)   

        logger.info(f"Classification Report:\n{resampled_report}")
        logger.info(f"Confusion Matrix:\n{resampled_cm}") 
        logger.info(f"Accuracy: {resampled_accuracy}")   
            
        # save the model
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.sm_model_name))
        joblib.dump(sm_preprocessor, os.path.join(self.config.root_dir, self.config.sm_processor_name))