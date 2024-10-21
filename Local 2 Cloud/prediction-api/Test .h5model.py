import tensorflow as tf
import numpy as np
import json

# Load the saved model without compiling it
model = tf.keras.models.load_model('california_housing_model.h5', compile=False)

# Compile the model with the same settings as before (if necessary)
model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# Example input JSON
input_json = '''
{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.023809,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}
'''

# Load JSON data
data = json.loads(input_json)

# Convert the JSON data to a numpy array
input_data = np.array([[data['MedInc'], data['HouseAge'], data['AveRooms'], data['AveBedrms'], 
                        data['Population'], data['AveOccup'], data['Latitude'], data['Longitude']]])

# Make a prediction
prediction = model.predict(input_data)

# Output the predicted value (e.g., median house value)
print(f"Predicted Median House Value: {prediction[0][0]}")
