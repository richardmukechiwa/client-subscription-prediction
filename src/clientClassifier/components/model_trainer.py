import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from clientClassifier import logger
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE    
from clientClassifier.entity.config_entity import ModelTrainerConfig
from imblearn.pipeline import Pipeline as ImbPipeline


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        logger.info("Training model")

        # Load train and test data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Separate features and target variable
        X_train = train_data.drop(columns=[self.config.target_column], axis=1)
        y_train = train_data[self.config.target_column]

        X_test = test_data.drop(columns=[self.config.target_column], axis=1)
        y_test = test_data[self.config.target_column]

        # Identify numerical and categorical feature columns
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

        # Apply Label Encoding to target variable
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # Preprocessing for features
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Now you can fit a model using preprocessor and encoded targets

        logger.info("Preprocessing and label encoding complete")

        
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
        
        #create Confusion Matrix Display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        disp.plot()
        plt.title("Logistic Regression Matrix without SMOTE")
        plt.show()
        
        
        
        
                    

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
        joblib.dump(label_encoder, os.path.join(self.config.root_dir, self.config.label_encoder_name))
          
        
        logger.info(f"Model saved at: {model_path}")
        logger.info(f"Preprocessor saved at: {preprocessor_path}")      
        
     
        
  
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
        
        # Apply Label Encoding to target variable
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

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

        #  Fitting model on resampled data
        model = LogisticRegression(
            class_weight=self.config.class_weight,
            max_iter=self.config.max_iter,
            penalty=self.config.penalty,
            C=self.config.C,
            solver = self.config.solver,
            random_state=self.config.random_state
        )
        
        # use the sm_pipeline to transform the data
        sm_pipeline = ImbPipeline([
            ('preprocessor', sm_preprocessor),
            ('smote', SMOTE(random_state=self.config.random_state)),
            ('classifier', model)
        ])

        sm_pipeline.fit(X_train, y_train)
            
        # make predictions on the test data 
        y_pred = sm_pipeline.predict(X_test)

        # evaluate the model
        resampled_report = classification_report(y_test, y_pred)
        resampled_cm = confusion_matrix(y_test, y_pred)   
        resampled_accuracy = accuracy_score(y_test, y_pred)
        
        #create Confusion Matrix Display
        cm_display = ConfusionMatrixDisplay(confusion_matrix=resampled_cm, display_labels=label_encoder.classes_)
        cm_display.plot()
        plt.title("Logistic Regression with Matrix SMOTE")
        plt.show()
        
                                            

        logger.info(f"Classification Report:\n{resampled_report}")
        logger.info(f"Confusion Matrix:\n{resampled_cm}") 
        logger.info(f"Accuracy: {resampled_accuracy}")   
            
    
        # Save the model and preprocessor
        joblib.dump(sm_pipeline,
                    os.path.join(self.config.root_dir, 
                    self.config.sm_model_pipeline_name))
