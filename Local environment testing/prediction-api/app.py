import os
from flask import Flask, request
from house_predictor import HousePredictor

app = Flask(__name__)
app.config["DEBUG"] = False

@app.route("/house_predictor", methods=['POST'])  # path of the endpoint. Only HTTP POST request
def predict_str():
    # Get the prediction input data in the message body as a JSON payload
    prediction_input = request.get_json()
    return dp.predict_single_record(prediction_input)

dp = HousePredictor()

# The code within this conditional block will only run the python file is executed as a script.
if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0', debug=False)
