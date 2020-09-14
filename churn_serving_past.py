import pickle
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

with open('churn-model-development.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


customer = {
    'customerid': '8879-zkjof',
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'no',
    'tenure': 41,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'no',
    'deviceprotection': 'yes',
    'techsupport': 'yes',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'one_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 79.85,
    'totalcharges': 3320.75,
}

prediction = predict_single(customer, dv, model)

print('prediction: %.3f' % prediction)

if prediction >= 0.5:
    print('verdict: Churn')
else:
    print('verdict: Not churn')
