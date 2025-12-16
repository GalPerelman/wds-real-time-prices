import pandas as pd
import numpy as np
import pytz
import datetime
from sklearn.impute import KNNImputer


def data_completion(data, n_neighbors=3):
    knn_impute = KNNImputer(n_neighbors=n_neighbors)
    data_imputed = knn_impute.fit_transform(data)
    data_imputed = pd.DataFrame(data_imputed, columns=data.columns, index=data.index)
    return data_imputed


def normalize_datetime(dt, target_tz='UTC'):
    """
    Normalize various datetime types to timezone-aware datetime.datetime in UTC.
    Handles:
    - datetime.datetime (naive or aware)
    - pd.Timestamp
    - np.datetime64
    """
    # Convert pd.Timestamp to datetime
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()

    # Convert np.datetime64 to datetime
    elif isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt).to_pydatetime()

    # Now we have a datetime.datetime object
    if not isinstance(dt, datetime.datetime):
        raise TypeError(f"Cannot convert {type(dt)} to datetime")

    # Handle timezone
    if dt.tzinfo is None:
        # If naive, assume UTC and make aware
        dt = dt.replace(tzinfo=pytz.UTC)

    # Convert to target timezone
    target = pytz.timezone(target_tz)
    return dt.astimezone(target)