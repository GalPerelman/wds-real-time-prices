import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils
import uncertainty_utils
from opt import BaseOptModel, RO, SP
from network import LPNetwork
import forecast


class MPC:
    def __init__(self, net_path, opt_method, db, uset_estimator, start_datetime, step_horizon, wait, experiment_horizon,
                 realizations, prices_col, demand_cols, forecast_method="xgb", forecast_params=None,
                 milp=True, holiday_country='HU', **kwargs):
        self.net_path = net_path
        self.opt_method = opt_method
        self.db = db
        self.uset_estimator = uset_estimator
        self.start_datetime = utils.normalize_datetime(start_datetime)
        self.step_horizon = step_horizon
        self.wait = wait
        self.experiment_horizon = experiment_horizon
        self.realizations = realizations
        self.forecast_method = forecast_method
        self.forecast_params = forecast_params or {}
        self.milp = milp

        # Forecast configuration
        self.prices_col = prices_col
        self.demand_cols = demand_cols
        self.holiday_country = holiday_country
        self.buffer = forecast.RealizationsBuffer(self.db.forecast_training_data)

        # different parameters for different optimization methods
        self.u_set = kwargs.get("uncertainty_set", "ellipsoid")
        self.omega = kwargs.get("omega", 1)  # for RO with Ellipsoidal Uncertainty
        self.gamma = kwargs.get("gamma", (2, 5))  # for RO with Budget of Uncertainty
        self.k = kwargs.get("k", 1.5)  # for RO with Budget of Uncertainty
        self.alpha = kwargs.get("alpha", (0.95, 0.80))  # for SP: (alpha_start, alpha_end)
        self.u_set = kwargs.get("uncertainty_set", "ellipsoid")  # for RO

        self.t_idx = 0  # counter of time steps from start
        self.current_time = self.start_datetime  # timestamp of the first step

        self.last_forecast = None
        self.net = None
        self.opt_model = None
        self.results = pd.DataFrame()

        self.forecaster = self.prepare_forecaster()
        self.predict_next_horizon()

    def init_opt_model(self):
        self.net = LPNetwork(self.net_path)
        if self.opt_method in ["lp", "pto"]:
            model = BaseOptModel(network=self.net, t=self.step_horizon, forecast=self.last_forecast,
                                 uset_estimator=self.uset_estimator, milp=self.milp)
        elif self.opt_method in ["ro", "RO"]:
            model = RO(network=self.net, t=self.step_horizon, omega=self.omega, gamma=self.gamma, k=self.k,
                       forecast=self.last_forecast, uset_estimator=self.uset_estimator, milp=self.milp,
                       uncertainty_set=self.u_set)
        elif self.opt_method in ["sp", "SP", "CC", "SAA"]:
            model = SP(network=self.net, t=self.step_horizon, alpha=self.alpha,
                       forecast=self.last_forecast, uset_estimator=self.uset_estimator, milp=self.milp)
        else:
            raise ValueError(f"Unknown optimization method: {self.opt_method}")

        self.opt_model = model

    def prepare_forecaster(self):
        f = forecast.Forecast(
            training_data=self.db.forecast_training_data,
            prices_col=self.prices_col,
            demand_cols=self.demand_cols,
            holiday_country=self.holiday_country,
            buffer=self.buffer
        )
        f.preprocess()

        f.fit(model_name=self.forecast_method, target="demand", **self.forecast_params)
        f.fit(model_name=self.forecast_method, target="price", **self.forecast_params)
        return f

    def update_parameters(self):
        """Push new Parameters (prices/demands/initial state, etc.) into the model."""
        pass

    def preprocess_next_horizon(self):
        next_horizon_end = self.current_time + pd.to_timedelta(f'{self.step_horizon}h')
        next_horizon = pd.DataFrame(index=pd.date_range(start=self.current_time, end=next_horizon_end, freq="h"))
        next_horizon_data = forecast.Forecast.construct_datetime_features(next_horizon, holiday_country=self.holiday_country)
        return next_horizon_data

    def predict_next_horizon(self):
        next_horizon_x = self.preprocess_next_horizon()
        demand_forecast = self.forecaster.predict(x=next_horizon_x, target="demand")
        price_forecast = self.forecaster.predict(x=next_horizon_x, target="price")
        self.last_forecast = pd.DataFrame(index=next_horizon_x.index, data=np.hstack([price_forecast, demand_forecast]),
                                          columns=[self.forecaster.prices_col] + self.forecaster.demand_cols)

    def run(self):
        for t in range(self.experiment_horizon // self.wait):
            if self.buffer.should_update(self.current_time):
                yesterday_end = self.current_time - pd.Timedelta(hours=1)
                yesterday_start = self.current_time - pd.Timedelta(hours=24)
                prev_day = self.realizations.loc[yesterday_start:yesterday_end]
                if not prev_day.empty:
                    self.buffer.update(prev_day)

            self.predict_next_horizon()
            self.opt_model.forecast = self.last_forecast
            self.opt_model.apply_forecast()
            self.opt_model.formulate(start_hour=self.current_time.hour)
            status, cost = self.opt_model.solve()

            step_end = self.current_time + datetime.timedelta(hours=self.wait - 1)
            realization = self.realizations.loc[self.current_time: step_end]
            step_tank_volumes = {}
            for i, (tank_id, tank_data) in enumerate(self.net.tanks.iterrows()):
                vol = self.net.get_tank_vol(t=self.wait, tank_id=tank_id,
                                            x_fsp=self.opt_model.x_fsp_value[:self.wait, :],
                                            x_vsp=self.opt_model.x_vsp_value[:self.wait, :],
                                            sample=realization.iloc[:self.wait, 1 + i].values)

                self.net.tanks.loc[tank_id, "init_vol"] = vol[-1]
                step_tank_volumes[tank_id] = vol

            # record results
            df = pd.DataFrame({
                **{"idx": np.repeat(self.t_idx, self.wait), "time": realization.index},
                **{f"x_fsp_{i}": self.opt_model.x_fsp_value[:self.wait, i] for i in range(self.net.n_fsp)},
                **{f"x_vsp_{i}": self.opt_model.x_vsp_value[:self.wait, i] for i in range(self.net.n_vsp)},
                **{f"predicted_tariff": self.last_forecast.loc[realization.index, 'tariff'].values[:self.wait]},
                **{f"realized_tariff": realization['tariff'].values[:self.wait]},
                **{f"predicted_demand_{tank_id}":
                       self.last_forecast.loc[realization.index, tank_data["demand"]].values[:self.wait]
                   for tank_id, tank_data in self.net.tanks.iterrows()},
                **{f"realized_demand_{tank_id}": realization.loc[:, tank_data["demand"]].values[:self.wait] for
                   tank_id, tank_data in
                   self.net.tanks.iterrows()},
                **{f"vol_tank_{tank_id}": step_tank_volumes[tank_id].flatten() for tank_id in self.net.tanks.index}
            },
            )

            self.results = pd.concat([self.results, df], ignore_index=True)
            self.t_idx += 1
            self.current_time = self.current_time + datetime.timedelta(hours=self.wait)

    def plot_results(self):
        fig, axes = plt.subplots(nrows=self.net.n_tanks + 1, sharex=True)
        for i, tank_id in enumerate(self.net.tanks.index):
            axes[i].plot(self.results[f"vol_tank_{tank_id}"], label=f"vol_tank_{tank_id}")
            axes[i].axhline(self.net.tanks.loc[tank_id, 'min_vol'], color='red', label='Min Volume' if i == 0 else "")
            axes[i].axhline(self.net.tanks.loc[tank_id, 'max_vol'], color='red', label='Max Volume' if i == 0 else "")
            axes[i].set_ylabel(f'Tank {tank_id}\nVolume\n($m^3$)')
        axes[-1].bar(range(len(self.results)), self.results['predicted_tariff'], alpha=0.5, edgecolor='k')
        axes[-1].set_ylabel('Predicted\nTariff\n(€/MWh)')
        axes[-1].set_xlabel('Time (Hr)')
        fig.align_ylabels()
        plt.subplots_adjust(left=0.18, right=0.95)

        fig, axes = plt.subplots(nrows=self.net.n_tanks + 1, sharex=True)
        axes[0].plot(self.results['realized_tariff'])
        axes[0].plot(self.results['predicted_tariff'])
        axes[0].set_ylabel('Tariff\n(€/MWh)')
        axes[0].text(0.01, 0.8, 'Electricity Tariff', transform=axes[0].transAxes)
        for i, tank_id in enumerate(self.net.tanks.index):
            axes[i + 1].plot(self.results[f"realized_demand_{tank_id}"], label=f"Realized")
            axes[i + 1].plot(self.results[f"predicted_demand_{tank_id}"], label=f"Predicted")
            axes[i + 1].set_ylabel(f'Demand\n($m^3/h$)')
            axes[i + 1].text(0.01, 0.8, f'Tank {tank_id} Demand', transform=axes[i + 1].transAxes)

        axes[-1].set_xlabel('Time (Hr)')
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.975), ncol=2)
        fig.align_ylabels()
        plt.subplots_adjust(left=0.15, right=0.95)