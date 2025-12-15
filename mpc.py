import datetime

import pandas as pd
import numpy as np

from opt import BaseOptModel, RO, SP
from network import LPNetwork
import forecast


class MPC:
    def __init__(self, net_path, opt_method, db, start_datetime, step_horizon, wait, experiment_horizon,
                 forecast_method="xgb", milp=True, **kwargs):
        self.net_path = net_path
        self.opt_method = opt_method
        self.db = db
        self.start_datetime = start_datetime
        self.step_horizon = step_horizon
        self.wait = wait
        self.experiment_horizon = experiment_horizon
        self.forecast_method = forecast_method
        self.milp = milp

        # different parameters for different optimization methods
        self.omega = kwargs.get("omega", 1)  # for RO with Ellipsoidal Uncertainty
        self.gamma = kwargs.get("gamma", (2, 5))  # for RO with Budget of Uncertainty
        self.epsilon = kwargs.get("epsilon", 0.05)  # for SP with Chance Constraints
        self.alpha = kwargs.get("alpha", 1.0)  # for SP - CC relaxation over time parameter

        self.t_idx = 0  # counter of time steps from start
        self.current_time = self.start_datetime  # timestamp of the first step

        self.forecaster = self.prepare_forecaster()
        self.last_forecast = self.predict_next_horizon()

        self.net = None
        self.opt_model = None

    def init_opt_model(self):
        self.net = LPNetwork(self.net_path)
        if self.opt_method in ["lp", "pto"]:
            model = BaseOptModel(network=self.net, t=self.step_horizon, forecast=self.last_forecast,
                                 database=self.db.data, milp=self.milp)
        elif self.opt_method in ["ro", "RO"]:
            model = RO(network=self.net, t=self.step_horizon, omega=self.omega, gamma=self.gamma,
                       forecast=self.last_forecast, database=self.db.data, milp=self.milp)
        elif self.opt_method in ["sp", "SP", "CC", "SAA"]:
            model = SP(network=self.net, t=self.step_horizon, epsilon=self.epsilon, alpha=self.alpha,
                       forecast=self.last_forecast, database=self.db.data, milp=self.milp)
        else:
            raise ValueError(f"Unknown optimization method: {self.opt_method}")

        self.opt_model = model

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

    def predict_next_horizon(self):
        next_horizon_x = self.preprocess_next_horizon()
        demand_forecast = self.forecaster.predict(x=next_horizon_x, target="demand")
        price_forecast = self.forecaster.predict(x=next_horizon_x, target="price")
        forecast = pd.DataFrame(index=next_horizon_x.index, data=np.hstack([price_forecast, demand_forecast]),
                                columns=[self.forecaster.prices_col] + self.forecaster.demand_cols)
        return forecast

    def run(self):
        for t in range(self.experiment_horizon):

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