import datetime

# from data
AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"
LOCAL_PATH = "/Users/R/code/rodrigomras/TaxiFareModel/raw_data/train.csv"
BUCKET_NAME = 'wagon-ml-sousa-566'
BUCKET_TRAIN_DATA_PATH = 'data/train.csv'
DIST_ARGS = dict(start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude")

# from predict
PATH_TO_LOCAL_MODEL = '/Users/R/code/rodrigomras/TaxiFareModel/model.joblib'

# form trainer
MODEL_DIRECTY = "PipelineTest"  # must the same as PATH_TO_MODEL inside Makefile
MLFLOW_URI = "https://mlflow.lewagon.co/"
#BUCKET_NAME = 'wagon-ml-sousa-566'
#BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'
MODEL_VERSION = f'V_{datetime.datetime.now()}'
MODEL_NAME = 'taxifare'
STORAGE_LOCATION = f'models/taxifare/{MODEL_VERSION}/model.joblib'

LOCAL_CACHE="/Users/R/rodrigomras/TaxifareModel/TaxifareModel/data"
