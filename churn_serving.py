import pickle
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify

app = Flask('churn')


def predict_single(customer, dv, model):
    numerical_features = ['tenure', 'monthlycharges', 'totalcharges']
    categorical_features = ['gender',
                             'seniorcitizen',
                             'partner',
                             'dependents',
                             'phoneservice',
                             'multiplelines',
                             'internetservice',
                             'onlinesecurity',
                             'onlinebackup',
                             'deviceprotection',
                             'techsupport',
                             'streamingtv',
                             'streamingmovies',
                             'contract',
                             'paperlessbilling',
                             'paymentmethod'
                             ]
    inp_data = defaultdict(list)
    for k in customer.keys():
        inp_data[k].append(customer[k])
    df = pd.DataFrame(inp_data)

    cat = df[categorical_features].to_dict(orient='rows')
    X_categorical = dv.transform(cat)
    sc = StandardScaler()
    sc.fit(df[numerical_features])
    X_numerical = sc.transform(df[numerical_features])

    X = np.concatenate((X_categorical, X_numerical), axis=1)

    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5

    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn),
    }

    return jsonify(result)


with open('churn-model-development.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

