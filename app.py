from flask import Flask, render_template, request
import pandas as pd
import os
from clientClassifier.pipeline.prediction import PredictionPipeline

app = Flask(__name__)


@app.route('/', methods=['GET'])
def homePage():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def training():
    os.system('python main.py')
    return 'Training successfully completed!'  

    

@app.route('/predict', methods=['GET', 'POST'])  # Allow both GET and POST
def index():
    if request.method == 'POST':
        try:
            # Get the input data from the form
            age = int(request.form['age'])
            month = str(request.form['month'])
            day = int(request.form['day'])
            balance = int(request.form['balance'])
            poutcome = str(request.form['poutcome'])

            # Create a DataFrame from the input data
            data = pd.DataFrame([[age, month, day, balance, poutcome]], 
                                columns=['age', 'month', 'day', 'balance', 'poutcome'])

            # Load the prediction pipeline and make a prediction
            pipeline = PredictionPipeline()
            predict = pipeline.predict(data)

            # Render the result on the template
            return render_template('results.html', prediction=str(predict))

        except Exception as e:  
            print("Error in prediction:", e)
            return 'Something went wrong! Please check your input data.'
    else:
        return render_template('index.html')  # Render form on GET request

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

    