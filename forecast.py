import datetime
import pytz
import pandas as pd
import numpy as np
from copy import deepcopy

import xgboost as xgb
import lightgbm as lgb
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor


class Forecast:
    def __init__(self, db, prices_col, demand_cols):
        self.db = db
        self.prices_col = prices_col
        self.demand_cols = demand_cols
        self.data = deepcopy(self.db.data)

        self.models = {"xgb": xgb.XGBRegressor, "lgb": lgb.LGBMRegressor, "rf": RandomForestRegressor}

    def preprocess(self):
        self.data_completion()
        self.construct_datetime_features()
        self.lag_features(cols_to_lag={**{col: 6 for col in self.demand_cols}, **{self.prices_col: 5}})
        self.data.dropna()

    def data_completion(self):
        knn_impute = KNNImputer(n_neighbors=3)
        data_imputed = knn_impute.fit_transform(self.data)
        data_imputed = pd.DataFrame(data_imputed, columns=self.data.columns, index=self.data.index)
        self.data = data_imputed

    def construct_datetime_features(self):
        """
        Function to add categories based on the datetime index: Month, weekday, hour, special dates
        """
        self.data['hour_sin'] = np.sin(self.data.index.hour * (2. * np.pi / 24))
        self.data['hour_cos'] = np.cos(self.data.index.hour * (2. * np.pi / 24))
        self.data['day_sin'] = np.sin(self.data.index.day * (2. * np.pi / 31))  # Assuming max 31 days in a month
        self.data['day_cos'] = np.cos(self.data.index.day * (2. * np.pi / 31))
        self.data['weekday_sin'] = np.sin(self.data.index.weekday * (2. * np.pi / 7))
        self.data['weekday_cos'] = np.cos(self.data.index.weekday * (2. * np.pi / 7))
        self.data['month_sin'] = np.sin(self.data.index.month * (2. * np.pi / 12))
        self.data['month_cos'] = np.cos(self.data.index.month * (2. * np.pi / 12))
        self.data['weeknum_sin'] = np.sin((self.data.index.strftime('%U').astype(int) + 1) * (2. * np.pi / 52))
        self.data['weeknum_cos'] = np.cos((self.data.index.strftime('%U').astype(int) + 1) * (2. * np.pi / 52))

        def is_dst(dt):
            return dt.dst() != pd.Timedelta(0)

        self.data['is_dst'] = self.data.index.map(is_dst).astype(int)
        self.data['is_special'] = self.data.index.normalize().isin(SpecialDates.dates).astype(int)

    def lag_features(self, cols_to_lag: dict):
        lagged_cols = []
        for label, lags in cols_to_lag.items():
            for lag in range(1, lags + 1):
                self.data[label + f'_{lag}'] = self.data[label].shift(lag)
                lagged_cols.append(label + f'_{lag}')

        return lagged_cols

    def data_split(self, target_col):
        x = self.data.drop([self.prices_col] + self.demand_cols, axis=1)
        y = self.data[[target_col]]
        return x, y

    def fit(self, model_name, target_col):
        reg = self.models[model_name]()
        x, y = self.data_split(target_col)
        reg.fit(x, y)
        return reg

    def predict(self, reg, x):
        return reg.predict(x)


class SpecialDates:
    TZ = pytz.timezone('CET')
    dates = [
        TZ.localize(datetime.datetime(2022, 1, 1)),  # New Year's
        TZ.localize(datetime.datetime(2022, 4, 15)),  # Easter
        TZ.localize(datetime.datetime(2022, 4, 16)),  # Easter
        TZ.localize(datetime.datetime(2022, 4, 17)),  # Easter
        TZ.localize(datetime.datetime(2022, 4, 18)),  # Easter
        TZ.localize(datetime.datetime(2022, 5, 1)),  # Labour Day
        TZ.localize(datetime.datetime(2022, 12, 25)),  # Christmas
        TZ.localize(datetime.datetime(2022, 12, 26)),  # Stephen's Day
    ]