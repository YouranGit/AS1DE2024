# importing Flask and other modules
import json
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
                "feature_8": float(request.form.get("feature_8"))
               
            }
        ]

        logging.debug("Prediction input : %s", prediction_input)

        # Hard-code the predictor API URL here
        predictor_api_url = "http://localhost:5000/house_predictor"  # Use the local API endpoint
        res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))

        prediction_value = res.json()['result']
        logging.info("Prediction Output : %s", prediction_value)
        return render_template("result.html",
                               prediction_variable=prediction_value)

    else:
        return jsonify(message="Method Not Allowed"), 405

# The code within this conditional block will only run the python file is executed as a script.
if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0', debug=False)
