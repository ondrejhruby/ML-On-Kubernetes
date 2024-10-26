import pickle
from flask import Flask, request, jsonify
import numpy as np

# Load the trained model
with open("iris_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Create a Flask application
app = Flask(__name__)

# Define a route for inference
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data["features"]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    species = int(prediction[0])
    
    # Return the result as JSON
    return jsonify({"species": species})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
