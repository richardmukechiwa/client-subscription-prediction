import joblib
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        # Define the paths to the model and preprocessor
        self.model_path = Path('artifacts/model_trainer/xgb.joblib')
       
        
        # Load the model and preprocessor
        self.model = joblib.load(self.model_path)
        
        
    def predict(self, data):
        
        # Check if the input is a DataFrame, dict, or list
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a DataFrame, dict, or list.")  
        
        # Ensure the DataFrame has the correct columns
        expected_columns = ['age', 'month', 'day', 'balance', 'poutcome']   
        if not all(col in data.columns for col in expected_columns):
            raise ValueError(f"Input DataFrame must contain the following columns: {expected_columns}") 
        
        # Apply the preprocessor to the input data
    
        
        # Make predictions using the loaded model
        prediction = self.model.predict(data)
        
        # Convert NumPy array to list for JSON response
        return prediction