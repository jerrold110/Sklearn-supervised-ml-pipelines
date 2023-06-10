import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

print(os.getcwd())
print(os.path.realpath(__file__))

# change working directory to current directory
path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)

# load data and model
x = pd.read_csv('data/x_test_data.csv')
y = pd.read_csv('data/y_test_data.csv')
y_true = y['survived']
model = joblib.load('model/classifier.joblib')

# predict data
predictions = model.predict(x)
print('classifier score: ',accuracy_score(y_true, predictions))