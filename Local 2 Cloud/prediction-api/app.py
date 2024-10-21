import os
from flask import Flask, request
from house_predictor import HousePredictor
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config["DEBUG"] = False

dp = HousePredictor()

@app.route("/house_predictor", methods=['POST'])  # path of the endpoint. Only HTTP POST request
def predict_str():
    # Get the prediction input data in the message body as a JSON payload
    prediction_input = request.get_json()
    logging.debug("Received input %s", prediction_input)
    return dp.predict_single_record(prediction_input)



# The code within this conditional block will only run the python file is executed as a script.
if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0', debug=False)
