# imports
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse

# import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """

        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        # distance and time pipelines
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder(handle_unknown='ignore'))

        # preprocessing pipeline
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']
        preproc_pipe = ColumnTransformer([('time', pipe_time, time_cols),
                                          ('distance', pipe_distance, dist_cols)])

        # full pipeline
        self.pipeline = Pipeline(steps=[
                                ('preprocessing', preproc_pipe),
                                ('model', LinearRegression())
                                    ])

        # return pipeline

    def run(self):
        """set and train the pipeline"""

        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""

        y_pred = self.pipeline.predict(X_test)
        self.rmse = compute_rmse(y_pred, y_test)
        # print(f'The RMSE of your model is {rmse}')

        return self.rmse


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # # set X and y
    X = df.drop(columns=['fare_amount'])
    y = df[['fare_amount']]
    # # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # # train
    trainer = Trainer(X_train, y_train)
    trainer.set_pipeline()
    trainer.run()
    # evaluate
    trainer.evaluate(X_test, y_test)
    print(f"Your model's RMSE is {trainer.rmse}")
