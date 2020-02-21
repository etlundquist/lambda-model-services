import json
import boto3
import pickle

import numpy as np
import pandas as pd

# load the saved model object from S3
# -----------------------------------

bucket_name = "etlundquist-us-west-2-models"
object_name = "lambda-example/regressor.pkl"

s3 = boto3.client('s3')
body = s3.get_object(Bucket=bucket_name, Key=object_name)['Body'].read()
model = pickle.loads(body)

# define the main lambda event handler
# -----------------------------------

def lambda_handler(event, context):
    """parse a feature vector from the event body and return the resulting model score"""

    params = json.loads(event['body'])
    fvector = np.array(params['fvector']).reshape(1, -1)
    score = model.predict(fvector)[0]

    response = {
        'statusCode': 200,
        'body': json.dumps(score)
    }
    return response
