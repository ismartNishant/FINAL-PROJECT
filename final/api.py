

from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import sys

# API endpoint URL would consist /predict
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict1():
    if pred:
        try:

            json_ = request.json
            #print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            #print("q1:"+ query)
            query = query.reindex(columns=model_columns, fill_value=0)
            #print("q2:" + query)
            prediction = list(pred.predict(query))

            return jsonify({'prediction': prediction})
        except:

            return jsonify({'trace': traceback.format_exc()})

    else:
        print('Train the model first')
        return ('No model here to use')
if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    pred = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)