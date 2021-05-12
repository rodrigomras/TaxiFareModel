from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from TaxiFareModel.params import PATH_TO_LOCAL_MODEL

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict_fare")
def predict(key,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count):

    user_input = {"key": key.replace('%',' '),
                "pickup_datetime": pickup_datetime.replace('%',' '),
                "pickup_longitude": float(pickup_longitude),
                "pickup_latitude": float(pickup_latitude),
                "dropoff_longitude": float(dropoff_longitude),
                "dropoff_latitude": float(dropoff_latitude),
                "passenger_count": float(passenger_count)
                }

    X_pred = pd.DataFrame([user_input])
    # print(X_pred)
    pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    y_pred = pipeline.predict(X_pred)
    print(type(y_pred[0]))

    return {'prediction': float(y_pred[0])}
