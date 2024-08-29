# prediction.py
#import joblib library

#import joblib

# Load the saved model
#model = joblib.load("rf_model.sav")

#define the function for  model prediction

#import joblib
#def predict(data):
    #clf = joblib.load("rf_model.sav")
    #return clf.predict(data)

import pickle

def predict(data):
    with open("rf_model.sav", "rb") as model_file:
        clf = pickle.load(model_file)
    return clf.predict(data)
