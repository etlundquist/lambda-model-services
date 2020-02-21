import os
import sys
import json
import boto3
import pickle
import requests
import subprocess

import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# load training data
# ------------------

boston = load_boston()
X, y = shuffle(boston.data, boston.target, random_state=1492)

# calculate naive baseline
# ------------------------

y_hat_0 = np.repeat(y.mean(), len(y))
mse_0 = mean_squared_error(y, y_hat_0)

# tune/train model via cross validation
# -------------------------------------

param_grid = {'n_estimators': [25, 50, 100], 'max_depth': [5, 10, 20], 'min_samples_leaf': [1, 10, 50]}
clf = RandomForestRegressor()
cv = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
cv.fit(X, y)

mse_1 = -cv.best_score_
best_params = cv.best_params_
best_model = cv.best_estimator_

print("null model MSE: {} best model MSE: {}".format(mse_0, mse_1))
print("best params: {}".format(best_params))

# serialize the trained model and save locally
# --------------------------------------------

model_root = "~/Repos/lambda-model-services/models"
model_path = os.path.join(model_root, "regressor.pkl")
model_file = open(model_path, "wb")
pickle.dump(best_model, model_file)

# upload the saved model object to S3
# -----------------------------------

bucket_name = "etlundquist-us-west-2-models"
object_name = "lambda-example/regressor.pkl"

session = boto3.Session(profile_name='etlundquist', region_name='us-west-2')
s3 = session.client(service_name='s3')
s3.upload_file(Filename=model_path, Bucket=bucket_name, Key=object_name)

# load the saved model object from S3
# -----------------------------------

body = s3.get_object(Bucket=bucket_name, Key=object_name)['Body'].read()
model = pickle.loads(body)

# format a JSON feature vector to score with the model
# ----------------------------------------------------

payload = json.dumps({'fvector': [3.613, 11.363, 11.136, 0.069, 0.554, 6.284, 68.574, 3.795, 9.549, 408.237, 18.455, 356.674, 12.653]})
fvector = np.array(json.loads(payload)['fvector']).reshape(1, -1)
score = model.predict(fvector)[0]

# call the model endpoint API with requests
# -----------------------------------------

endpoint = "https://btnhlt9bw0.execute-api.us-west-2.amazonaws.com/beta"
payload = {'fvector': [3.613, 11.363, 11.136, 0.069, 0.554, 6.284, 68.574, 3.795, 9.549, 408.237, 18.455, 356.674, 12.653]}
response = requests.post(endpoint, json=payload)

response.raise_for_status()
score = response.json()
print("model score: {}".format(score))

cmd = """curl -X POST -d "{\\"fvector\\": [3.613, 11.363, 11.136, 0.069, 0.554, 6.284, 68.574, 3.795, 9.549, 408.237, 18.455, 356.674, 12.653]}" https://btnhlt9bw0.execute-api.us-west-2.amazonaws.com/beta"""
res = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE)
float(res.stdout.decode('utf8'))

