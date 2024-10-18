# importing Flask and other modules
import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

# Flask constructor
app = Flask(__name__)


# A decorator used to tell the application
# which URL is associated function
@app.route("/checkprice", methods=["GET", "POST"])
def check_price():
    if request.method == "GET":
        return render_template("index.html")

    elif request.method == "POST":
        prediction_input = [
            {
                "feature_1": float(request.form.get("feature_1")),
                "feature_2": float(request.form.get("feature_2")),
                "feature_3": float(request.form.get("feature_3")),
                "feature_4": float(request.form.get("feature_4")),
                "feature_5": float(request.form.get("feature_5")),
                "feature_6": float(request.form.get("feature_6")),
                "feature_7": float(request.form.get("feature_7")),
                "feature_8": float(request.form.get("feature_8")),
                "feature_9": float(request.form.get("feature_9")),
                "feature_10": float(request.form.get("feature_10")),
                "feature_11": float(request.form.get("feature_11")),
                "feature_12": float(request.form.get("feature_12")),
                "feature_13": float(request.form.get("feature_13"))

            }
        ]

        logging.debug("Prediction input : %s", prediction_input)

        # use requests library to execute the prediction service API by sending an HTTP POST request
        # use an environment variable to find the value of the diabetes prediction API
        # json.dumps() function will convert a subset of Python objects into a json string.
        # json.loads() method can be used to parse a valid JSON string and convert it into a Python Dictionary.
        
        predictor_api_url = os.environ['PREDICTOR_API']
        res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))

        prediction_value = res.json()['result']
        logging.info("Prediction Output : %s", prediction_value)
        return render_template("result.html",
                               prediction_variable=prediction_value)

    else:
        return jsonify(message="Method Not Allowed"), 405  # The 405 Method Not Allowed should be used to indicate
    # that our app that does not allow the users to perform any other HTTP method (e.g., PUT and  DELETE) for
    # '/checkdiabetes' path


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=False)
