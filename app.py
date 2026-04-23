from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('models/diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        values = [float(x) for x in request.form.values()]

        # Convert to array
        final_input = np.array([values])

        # Scale input
        final_input_scaled = scaler.transform(final_input)

        # Prediction
        prediction = model.predict(final_input_scaled)

        if prediction[0] == 1:
            result = "Patient is likely Diabetic"
        else:
            result = "Patient is NOT Diabetic"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str(e))


if __name__ == "__main__":
    app.run(debug=True)