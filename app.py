from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the winning model
model = joblib.load('aqi_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Capture the 4 inputs from the form
    try:
        input_features = [float(x) for x in request.form.values()]
        features_array = [np.array(input_features)]
        
        # Get the prediction label (e.g., 'Poor')
        prediction = model.predict(features_array)
        result = prediction[0]
        
        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text="Error: Please enter valid numbers.")

if __name__ == "__main__":
    app.run(debug=True)