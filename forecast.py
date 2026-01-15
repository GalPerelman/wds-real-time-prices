import datetime
import pytz
import pandas as pd
import numpy as np
from copy import deepcopy

import holidays
import xgboost as xgb
import lightgbm as lgb
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor


def get_holidays(years, country='HU', tz=None):
    """
    Get holiday dates for specified years and country.

    Args:
        years: Single year (int) or list of years
        country: ISO country code (default 'HU' for Hungary). See holidays library for options.
        tz: Timezone (default CET). If None, returns timezone-naive dates.

    Returns:
        List of datetime objects for all holidays in the specified years.
    """
    if isinstance(years, int):
        years = [years]

    country_holidays = holidays.country_holidays(country, years=years)
    holiday_dates = list(country_holidays.keys())

    if tz is not None:
        holiday_dates = [tz.localize(datetime.datetime.combine(d, datetime.time())) for d in holiday_dates]

    return holiday_dates


class Forecast:
    def __init__(self, training_data, prices_col, demand_cols, holiday_country='HU'):
        """
        Initialize forecaster with training data.

        Args:
            training_data: DataFrame with datetime index containing prices and demands
            prices_col: Column name for electricity prices (e.g., 'tariff')
            demand_cols: List of column names for demands (e.g., ['a', 'b', 'c'])
            holiday_country: ISO country code for holidays (default 'HU'). Set to None to disable.
        """
        self.prices_col = prices_col
        self.demand_cols = demand_cols
        self.data = deepcopy(training_data)
        self.holiday_country = holiday_country

        self.models = {"xgb": xgb.XGBRegressor, "rf": RandomForestRegressor}

        self.price_model = None
        self.demand_model = None

    def preprocess(self):
        self.data_completion()
        self.data = self.construct_datetime_features(self.data, holiday_country=self.holiday_country)
        # self.lag_features(cols_to_lag={**{col: 6 for col in self.demand_cols}, **{self.prices_col: 5}})
        self.data.dropna()

    def data_completion(self):
        knn_impute = KNNImputer(n_neighbors=3)
        data_imputed = knn_impute.fit_transform(self.data)
        data_imputed = pd.DataFrame(data_imputed, columns=self.data.columns, index=self.data.index)
        self.data = data_imputed

    @staticmethod
    def construct_datetime_features(data, holiday_country='HU'):
        """
        Add datetime-based features to data.

        Args:
            data: DataFrame with datetime index
            holiday_country: ISO country code for holidays (default 'HU'). Set to None to disable.

        This function is a static method to allow preprocessing test data during MPC.
        """
        data['hour_sin'] = np.sin(data.index.hour * (2. * np.pi / 24))
        data['hour_cos'] = np.cos(data.index.hour * (2. * np.pi / 24))
        data['day_sin'] = np.sin(data.index.day * (2. * np.pi / 31))  # Assuming max 31 days in a month
        data['day_cos'] = np.cos(data.index.day * (2. * np.pi / 31))
        data['weekday_sin'] = np.sin(data.index.weekday * (2. * np.pi / 7))
        data['weekday_cos'] = np.cos(data.index.weekday * (2. * np.pi / 7))
        data['month_sin'] = np.sin(data.index.month * (2. * np.pi / 12))
        data['month_cos'] = np.cos(data.index.month * (2. * np.pi / 12))
        data['weeknum_sin'] = np.sin((data.index.strftime('%U').astype(int) + 1) * (2. * np.pi / 52))
        data['weeknum_cos'] = np.cos((data.index.strftime('%U').astype(int) + 1) * (2. * np.pi / 52))

        def is_dst(dt):
            return dt.dst() != pd.Timedelta(0)

        data['is_dst'] = data.index.map(is_dst).astype(int)

        # Holiday feature
        if holiday_country is not None:
            years = data.index.year.unique().tolist()
            holiday_dates = get_holidays(years, country=holiday_country)
            # Convert to same dtype as data index to avoid FutureWarning
            normalized_index = data.index.normalize()
            if normalized_index.tz is not None:
                # Data is timezone-aware, convert holiday_dates to match
                holiday_dates = pd.to_datetime(holiday_dates).tz_localize(normalized_index.tz)
            else:
                holiday_dates = pd.to_datetime(holiday_dates)
            data['is_special'] = normalized_index.isin(holiday_dates).astype(int)
        else:
            data['is_special'] = 0

        return data

    def lag_features(self, cols_to_lag: dict):
        lagged_cols = []
        for label, lags in cols_to_lag.items():
            for lag in range(1, lags + 1):
                self.data[label + f'_{lag}'] = self.data[label].shift(lag)
                lagged_cols.append(label + f'_{lag}')

        return lagged_cols

    def data_split(self, target_cols):
        x = self.data.drop([self.prices_col] + self.demand_cols, axis=1)
        y = self.data[target_cols]
        return x, y

    def fit(self, model_name, target):
        if target == "demand":
            reg = self.models[model_name]()
            x, y = self.data_split(self.demand_cols)
            reg.fit(x, y)
            self.demand_model = reg
        if target == "price":
            reg = self.models[model_name]()
            x, y = self.data_split([self.prices_col])
            reg.fit(x, y.values.ravel())  # Use ravel() to convert column vector to 1D array (sklearn expects 1D)
            self.price_model = reg

    def predict(self, x, target):
        if target == "demand":
            pred = self.demand_model.predict(x)
        if target == "price":
            pred = self.price_model.predict(x)

        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)

        return pred