# prediction.py
#import joblib library

import joblib

# Load the saved model
#model = joblib.load("rf_model.sav")

#define the function for  model prediction

def predict(data):
    clf = joblib.load(“rf_model.sav”)
    return clf.predict(data)
