import joblib
import pandas as pd
import numpy as np
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/xgb.joblib'))
        self.processor = joblib.load(Path('artifacts/model_trainer/xgb_preprocessor.joblib'))
        
        
    def predict(self, data):
        data_processing  = self.processor.transform(data)
        prediction = self.model.predict(data_processing)
        
        
        return prediction