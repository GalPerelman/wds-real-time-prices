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


class RealizationsBuffer:
    """
    Buffer for storing realized values, updated daily at midnight.

    Designed for AMR/AMI systems where meter readings are available daily.
    This is a data management class - it stores and retrieves data not tied to any specific forecasting method
    """
    def __init__(self, initial_data):
        self.data = initial_data.copy()
        self.last_update_date = self.data.index[-1].normalize()

    def should_update(self, current_time):
        """
        Check if buffer should be updated (at midnight).
        True if current_time is at midnight and we haven't updated today.
        """
        current_date = pd.Timestamp(current_time).normalize()
        is_midnight = pd.Timestamp(current_time).hour == 0
        is_new_day = current_date > self.last_update_date
        return is_midnight and is_new_day

    def update(self, new_realizations):
        """
        Update buffer with new realized values (typically previous day's data).
        """
        # Append new data
        self.data = pd.concat([self.data, new_realizations])

        # Remove duplicates (keep last in case of overlap)
        self.data = self.data[~self.data.index.duplicated(keep='last')]

        # Sort by index
        self.data = self.data.sort_index()

        # Update last update date
        self.last_update_date = new_realizations.index[-1].normalize()

    def get_values(self, times):
        """
        Get values at multiple times.

        Args:
            times: List of datetimes to look up.

        Returns:
            DataFrame with values at those times (missing times excluded).
        """
        available = [t for t in times if pd.Timestamp(t) in self.data.index]
        if not available:
            return None
        return self.data.loc[available]
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
    def __init__(self, training_data, prices_col, demand_cols, holiday_country='HU', buffer=None, random_state=42):
        """
        Initialize forecaster with training data.

        Args:
            training_data: DataFrame with datetime index containing prices and demands
            prices_col: Column name for electricity prices (e.g., 'tariff')
            demand_cols: List of column names for demands (e.g., ['a', 'b', 'c'])
            holiday_country: ISO country code for holidays (default 'HU'). Set to None to disable.
            buffer: RealizationsBuffer instance (required for naive method).
            random_state: Random seed for ML models (XGBoost, RandomForest) for reproducibility.
        """
        self.prices_col = prices_col
        self.demand_cols = demand_cols
        self.data = deepcopy(training_data)
        self.holiday_country = holiday_country
        self.buffer = buffer
        self.random_state = random_state

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
            reg = self.models[model_name](random_state=self.random_state)
            x, y = self.data_split(self.demand_cols)
            reg.fit(x, y)
            self.demand_model = reg
        if target == "price":
            reg = self.models[model_name](random_state=self.random_state)
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

    def generate_forecast(self, start_date, end_date):
        """
        Generate forecast for a date range.

        Args:
            start_date: Start of forecast horizon (string or datetime)
            end_date: End of forecast horizon (string or datetime)

        Returns:
            DataFrame with datetime index and columns: [prices_col] + demand_cols
            Ready to use as input to optimization models.
        """
        # Create date range for forecast horizon
        forecast_index = pd.date_range(start=start_date, end=end_date, freq='h')

        # Match timezone of training data if it has one
        if self.data.index.tz is not None:
            forecast_index = forecast_index.tz_localize(self.data.index.tz)

        forecast_df = pd.DataFrame(index=forecast_index)

        # Add datetime features
        forecast_features = self.construct_datetime_features(
            forecast_df, holiday_country=self.holiday_country
        )

        # Generate predictions
        price_pred = self.predict(forecast_features, target="price")
        demand_pred = self.predict(forecast_features, target="demand")

        # Combine into output DataFrame
        result = pd.DataFrame(
            index=forecast_index,
            data=np.hstack([price_pred, demand_pred]),
            columns=[self.prices_col] + self.demand_cols
        )

        return result