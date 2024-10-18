import json
import pandas as pd
from flask import jsonify
from keras.models import load_model
from keras.metrics import MeanSquaredError
import logging
from io import StringIO

class HousePredictor:
    def __init__(self):
        self.model = None

    def predict_single_record(self, prediction_input):
        logging.debug(prediction_input)
        if self.model is None:
            self.model = load_model('california_housing_model.h5', custom_objects={'MeanSquaredError': MeanSquaredError()})

        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')
        y_pred = self.model.predict(df)
        logging.info(y_pred[0])
        # Return the prediction outcome as a JSON message
        return jsonify({'result': str(y_pred[0])}), 200
