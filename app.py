from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd 
from  clientClassifier.pipeline.prediction import PredictionPipeline


app = Flask(__name__)


@app.route('/', methods=['GET']) # route to display the home page
def index():
    if request.method == 'POST':
        try:
            # Get the input data from the form
            age = int(request.form['age'])
            month = int(request.form['month'])
            day = int(request.form['day'])
            balance = int(request.form['balance'])
            poutcome = str(request.form['poutcome'])

            # Create a DataFrame from the input data
            data = pd.DataFrame([[age, month, day, balance, poutcome]], columns=['age', 'month', 'day', 'balance', 'poutcome'])

            # Load the prediction pipeline and make a prediction
            pipeline = PredictionPipeline()
            prediction = pipeline.predict(data)

            # Render the result on the template
            return render_template('index.html', prediction=prediction[0])
        except Exception as e:  
            return render_template('index.html', prediction="Error: " + str(e))
    else:
        return render_template('index.html')        

       
            

if __name__  == "__main__":
    app.run(host="0.0.0.0", port  = 8080)
    