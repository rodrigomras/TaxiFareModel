import pandas as pd

from TaxiFareModel.utils import simple_time_tracker
from TaxiFareModel.params import AWS_BUCKET_PATH, LOCAL_PATH, BUCKET_NAME, \
                                BUCKET_TRAIN_DATA_PATH, DIST_ARGS

@simple_time_tracker
def get_data(nrows=10000, local=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    if local:
        path = LOCAL_PATH
    else:
        path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
    df = pd.read_csv(path, nrows=nrows)
    if kwargs.get('optimize', False):
        df = df_optimized(df, verbose=True)
    return df

def download_model(model_directory="PipelineTest", bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model

def clean_df(df, test=False):
    """ Cleaning Data based on Kaggle test sample
    - remove high fare amount data points
    - keep samples where coordinate wihtin test range
    """
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def df_optimized(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numeircal columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return: df optimized
    """
    in_size = df.memory_usage(index=True).sum()
    # Optimized size here
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df


if __name__ == "__main__":
    params = dict(nrows=1000,
                  local=True, # set to False to get data from GCP (Storage or BigQuery)
                  optimize=True
                  )

    df = get_data(**params)

    print('shape: {}'.format(df.shape))
    print("size: {} Mb".format(df.memory_usage().sum() / 1e6))
