import joblib
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model_path = Path('artifacts/model_trainer/xgb_pipe.joblib')
        self.label_path = Path('artifacts/model_trainer/label_encoder.joblib')
  
        
        self.model  = joblib.load(self.model_path)
        self.label_encoder = joblib.load(self.label_path)
        
        
    def predict(self, data):
        
        # Check if the input is a DataFrame, dict, or list
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a DataFrame, dict, or list.")  
        
        # Apply the same preprocessing steps as during training
       
        prediction = self.model.predict(data)
        
        # Convert NumPy array to list for JSON response
        return self.label_encoder.inverse_transform(prediction).tolist()