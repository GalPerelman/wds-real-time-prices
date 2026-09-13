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


class NaiveModel:
    """
    Naive forecast model following sklearn estimator interface.

    For each time t, prediction = mean(value[t-168], value[t-336], ...)
    Averages the same hour/weekday from past n_weeks.
    """
    def __init__(self, buffer, n_weeks=1):
        """
        Initialize naive model.

        Args:
            buffer: RealizationsBuffer with historical data.
            n_weeks: Number of weeks to average over (default 1 = weekly naive).
        """
        self.buffer = buffer
        self.n_weeks = n_weeks
        self.target_cols = None

    def fit(self, X, y):
        """
        Fit method - infers target columns from y.
        Args:
            X: Feature matrix (ignored).
            y: Target values (DataFrame or Series with column names).
        """
        if hasattr(y, 'columns'):
            self.target_cols = y.columns.tolist()
        elif hasattr(y, 'name') and y.name is not None:
            self.target_cols = [y.name]
        else:
            raise ValueError("y must be a DataFrame or Series with column/name attribute")
        return self

    def predict(self, X):
        """
        Predict using average of past n_weeks at same hour/weekday.

        Args:
            X: DataFrame with datetime index indicating times to predict.

        Returns:
            ndarray of shape (n_samples, n_targets) with predictions.
        """
        if self.target_cols is None:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = []

        for target_time in X.index:
            target_time = pd.Timestamp(target_time)
            lag_times = [target_time - pd.Timedelta(hours=168 * week) for week in range(1, self.n_weeks + 1)]
            lag_values = self.buffer.get_values(lag_times)

            if lag_values is not None and len(lag_values) > 0:
                pred_row = lag_values[self.target_cols].mean().values
            else:
                # Fallback: use most recent available value
                pred_row = self.buffer.data[self.target_cols].iloc[-1].values

            predictions.append(pred_row)

        return np.array(predictions)


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

        self.models = {"xgb": xgb.XGBRegressor, "rf": RandomForestRegressor, "naive": NaiveModel}

        self.price_model = None
        self.demand_model = None
        self.n_weeks = 0

    def preprocess(self, n_weeks=0):
        """
        Args:
            n_weeks: number of week-multiple (168h, 336h, ...) lag columns to add per target col (xgb/rf).
                Same parameter/value as NaiveModel's n_weeks. For 'naive' it's consumed directly as the averaging
                window in fit() and this preprocessing step has no effect on its predictions.
        """
        self.n_weeks = n_weeks
        self.data_completion()
        self.data = self.construct_datetime_features(self.data, holiday_country=self.holiday_country)
        if self.n_weeks:
            self.lag_features(cols_to_lag={
                col: [168 * week for week in range(1, self.n_weeks + 1)]
                for col in [self.prices_col] + self.demand_cols
            })
        self.data = self.data.dropna()

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
        """
        Add lag columns to self.data (row-shift based - assumes a gapless hourly index).
        cols_to_lag: {column: [lag_hours, ...]}. Each lag column is named '{column}_lag{lag}h'.
        """
        lagged_cols = []
        for label, lags in cols_to_lag.items():
            for lag in lags:
                col_name = f'{label}_lag{lag}h'
                self.data[col_name] = self.data[label].shift(lag)
                lagged_cols.append(col_name)

        return lagged_cols

    def get_week_lag_values(self, index, n_weeks):
        """
        Vectorized week-multiple (168h, 336h, ...) lag lookup from self.buffer for a batch of
        target times, matching the column naming used by lag_features()/preprocess().
        Any 168h-multiple lag is guaranteed to be in the buffer regardless of what hour of day it was last updated
        """
        cols = [self.prices_col] + self.demand_cols
        lag_data = {}
        for col in cols:
            for week in range(1, n_weeks + 1):
                lag = 168 * week
                lag_data[f'{col}_lag{lag}h'] = self.buffer.data[col].reindex(index - pd.Timedelta(hours=lag)).values
        return pd.DataFrame(lag_data, index=index)

    def data_split(self, target_cols):
        x = self.data.drop([self.prices_col] + self.demand_cols, axis=1)
        y = self.data[target_cols]
        return x, y

    def fit(self, model_name, target, **model_params):
        """
        Fit a forecasting model.

        Args:
            model_name: Model type ('xgb', 'rf', 'naive').
            target: What to predict ('demand' or 'price').
            **model_params: Model-specific hyperparameters, e.g.:
                - naive: n_weeks=4 (averaging window)
                - xgb/rf: random_state=42, n_estimators=100, max_depth=5, etc.
                  n_weeks here is consumed by preprocess(), not passed to the regressor - see preprocess()'s docstring.
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")

        # Create model instance with appropriate parameters
        if model_name == 'naive':
            if self.buffer is None:
                raise ValueError("Naive model requires buffer. Pass buffer to Forecast.__init__().")
            reg = self.models[model_name](buffer=self.buffer, **model_params)
        else:
            # n_weeks controls lag feature generation in preprocess(), not a regressor kwarg
            model_params = {k: v for k, v in model_params.items() if k != 'n_weeks'}
            # Use self.random_state as default for reproducibility
            if 'random_state' not in model_params:
                model_params = {**model_params, 'random_state': self.random_state}
            reg = self.models[model_name](**model_params)

        # Fit model
        x, y = self.data_split(self.demand_cols if target == "demand" else [self.prices_col])

        if target == "demand":
            reg.fit(x, y)
            self.demand_model = reg
        elif target == "price":
            # For sklearn models, use ravel() for single target
            if model_name != 'naive':
                reg.fit(x, y.values.ravel())
            else:
                reg.fit(x, y)
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
        forecast_features = self.construct_datetime_features(forecast_df, holiday_country=self.holiday_country)
        if self.n_weeks:
            forecast_features = pd.concat(
                [forecast_features, self.get_week_lag_values(forecast_features.index, self.n_weeks)], axis=1
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