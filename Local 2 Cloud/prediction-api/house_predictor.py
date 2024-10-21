from keras.models import load_model
from keras.metrics import MeanSquaredError
from flask import jsonify
import json
import keras.saving
import pandas as pd
from io import StringIO

import keras.saving

# Register the 'mse' alias
@keras.saving.register_keras_serializable()
def mse():
    return MeanSquaredError()

class HousePredictor:
    def __init__(self):
        self.model = None

    def predict_single_record(self, prediction_input):
        if self.model is None:
            # Load the model and provide MeanSquaredError as a custom object
            self.model = load_model('california_housing_model.h5', custom_objects={'mse': mse})
        
        if isinstance(prediction_input, dict): 
            prediction_input = [prediction_input]
        # Ensure avoiding scalar value errors
        
        # Prepare the data and make predictions
        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')
        y_pred = self.model.predict(df)
        return jsonify({'result': str(y_pred[0])}), 200
