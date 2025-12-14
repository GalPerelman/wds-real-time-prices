import datetime

import pandas as pd
import numpy as np

import forecast


class MPC:
    def __init__(self, opt_model, db, start_datetime, step_horizon, wait, experiment_horizon, forecast_method="xgb"):
        self.opt_model = opt_model
        self.db = db
        self.start_datetime = datetime.datetime(start_datetime)
        self.step_horizon = step_horizon
        self.wait = wait
        self.experiment_horizon = experiment_horizon
        self.forecast_method = forecast_method

        self.t_idx = 0  # counter of time steps from start
        self.current_time = self.start_datetime  # timestamp of the first step

        self.forecaster = self.prepare_forecaster()

    def prepare_forecaster(self):
        f = forecast.Forecast(db=self.db, prices_col="tariff", demand_cols=['a', 'b', 'c'])
        f.preprocess()

        f.fit(model_name=self.forecast_method, target="demand")
        f.fit(model_name=self.forecast_method, target="price")
        return f

    def update_parameters(self):
        """Push new Parameters (prices/demands/initial state, etc.) into the model."""
        pass

    def preprocess_next_horizon(self):
        next_horizon_end = self.current_time + pd.to_timedelta(f'{self.step_horizon}h')
        next_horizon = pd.DataFrame(index=pd.date_range(start=self.current_time, end=next_horizon_end, freq="h"))
        next_horizon_data = forecast.Forecast.construct_datetime_features(next_horizon)
        return next_horizon_data

    def run(self):
        for t in range(self.experiment_horizon):
            next_horizon_x = self.preprocess_next_horizon()
            demand_forecast = self.forecaster.predict(x=next_horizon_x, target="demand")
            print(demand_forecast)

            self.t_idx += 1
            self.current_time = self.current_time + datetime.timedelta(hours=self.step_horizon)

    def apply_policy(self):
        pass


class PlantSimulator():
    """Obtain the 'real' system by one step given an action"""
    def __init__(self):
        pass

    def step(self):
        pass