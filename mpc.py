
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

import utils
import uncertainty_utils
from opt import BaseOptModel, RO, SP
from network import LPNetwork
import forecast


class MPC:
    def __init__(self, net_path, opt_method, db, uset_estimator, start_datetime, step_horizon, wait, experiment_horizon,
                 realizations, prices_col, demand_cols, forecast_method, milp, holiday_country, final_vol_constraint,
                 n_binary_steps=None, forecast_params=None, simulator=None, **kwargs):
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
        self.n_binary_steps = n_binary_steps

        # Forecast configuration
        self.prices_col = prices_col
        self.demand_cols = demand_cols
        self.holiday_country = holiday_country
        self.buffer = forecast.RealizationsBuffer(self.db.forecast_training_data)

        # different parameters for different optimization methods
        self.u_set = kwargs.get("uncertainty_set", None)  # required for RO; None errors loudly if unset
        self.omega = kwargs.get("omega", None)  # for RO with Ellipsoidal Uncertainty
        self.gamma = kwargs.get("gamma", None)  # for RO with Budget of Uncertainty
        self.k = kwargs.get("k", 1.5)  # for RO with Budget of Uncertainty
        self.alpha = kwargs.get("alpha", None)  # required for SP: (alpha_start, alpha_end); None errors loudly if unset
        self.sp_n_samples = kwargs.get("sp_n_samples", 1000)  # for SP: number of SAA samples
        self.sp_sampling_method = kwargs.get("sampling_method", "normal")  # for SP: scenario distribution
        self.lambda_cvar = kwargs.get("lambda_cvar", 0.0)  # for SP: CVaR risk weight (0 = pure SAA)
        self.solver_params = kwargs.get("solver_params", None)

        self.t_idx = 0  # counter of time steps from start
        self.current_time = self.start_datetime  # timestamp of the first step

        self.last_forecast = None
        self.net = None
        self.opt_model = None
        self.results = pd.DataFrame()
        self.n_infeasible = 0  # count of iterations where solver failed to find optimal solution

        self.simulator = simulator
        self.final_vol_constraint = final_vol_constraint

        self.latest_x_fsp = None  # fallback: full-horizon FSP schedule from last successful solve
        self.latest_x_vsp = None  # fallback: full-horizon VSP schedule from last successful solve
        self._fallback_offset = 0  # steps consumed from latest solution when consecutive failures occur

        self.forecaster = self.prepare_forecaster()
        self.predict_next_horizon()

    def init_opt_model(self):
        self.net = LPNetwork(self.net_path)
        if self.opt_method in ["lp", "pto"]:
            model = BaseOptModel(network=self.net, t=self.step_horizon, forecast=self.last_forecast,
                                 uset_estimator=self.uset_estimator, milp=self.milp,
                                 final_vol_constraint=self.final_vol_constraint,
                                 n_binary_steps=self.n_binary_steps,
                                 solver_params=self.solver_params)
        elif self.opt_method in ["ro", "RO"]:
            model = RO(network=self.net, t=self.step_horizon, omega=self.omega, gamma=self.gamma, k=self.k,
                       forecast=self.last_forecast, uset_estimator=self.uset_estimator, milp=self.milp,
                       uncertainty_set=self.u_set, final_vol_constraint=self.final_vol_constraint,
                       n_binary_steps=self.n_binary_steps,
                       solver_params=self.solver_params)
        elif self.opt_method in ["sp", "SP", "CC", "SAA"]:
            model = SP(network=self.net, t=self.step_horizon, alpha=self.alpha, lambda_cvar=self.lambda_cvar,
                       forecast=self.last_forecast, uset_estimator=self.uset_estimator, milp=self.milp,
                       final_vol_constraint=self.final_vol_constraint, n=self.sp_n_samples,
                       n_binary_steps=self.n_binary_steps,
                       sampling_method=self.sp_sampling_method, solver_params=self.solver_params)
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

    def preprocess_next_horizon(self):
        next_horizon_end = self.current_time + pd.to_timedelta(f'{self.step_horizon}h')
        next_horizon = pd.DataFrame(index=pd.date_range(start=self.current_time, end=next_horizon_end, freq="h"))
        next_horizon_data = forecast.Forecast.construct_datetime_features(next_horizon,
                                                                          holiday_country=self.holiday_country)
        return next_horizon_data

    def predict_next_horizon(self):
        next_horizon_x = self.preprocess_next_horizon()
        demand_forecast = self.forecaster.predict(x=next_horizon_x, target="demand")
        price_forecast = self.forecaster.predict(x=next_horizon_x, target="price")
        self.last_forecast = pd.DataFrame(index=next_horizon_x.index, data=np.hstack([price_forecast, demand_forecast]),
                                          columns=[self.forecaster.prices_col] + self.forecaster.demand_cols)

    def run(self):
        if self.simulator is not None:
            self.simulator.initialize(self.realizations)

        n_steps = self.experiment_horizon // self.wait
        _start_iloc = self.realizations.index.searchsorted(self.current_time)
        bar = tqdm(range(n_steps), desc="MPC", unit="step", ncols=180)
        for t in bar:
            step_iloc = _start_iloc + t * self.wait
            self.current_time = self.realizations.index[step_iloc]

            if self.buffer.should_update(self.current_time):
                prev_day = self.realizations.iloc[max(0, step_iloc - 24): step_iloc]
                if not prev_day.empty:
                    self.buffer.update(prev_day)

            self.predict_next_horizon()
            self.opt_model.forecast = self.last_forecast
            self.opt_model.apply_forecast()
            self.opt_model.formulate(start_hour=self.current_time.hour)
            status, cost = self.opt_model.solve()
            solver_time = self.opt_model.solve_time
            solver_mip_gap = self.opt_model.solver_mip_gap

            if status not in cp.settings.SOLUTION_PRESENT:
                self.n_infeasible += 1
                can_fallback = (self.latest_x_fsp is not None and
                                self._fallback_offset + self.wait <= self.step_horizon)
                if not can_fallback:
                    self.t_idx += 1
                    bar.set_postfix_str(
                        f"{self.current_time.strftime('%m-%d %H:%M')} | {status} | no fallback | infeasible: {self.n_infeasible}"
                    )
                    continue
                x_fsp = self.latest_x_fsp[self._fallback_offset:self._fallback_offset + self.wait, :]
                x_vsp = self.latest_x_vsp[self._fallback_offset:self._fallback_offset + self.wait, :]
                self._fallback_offset += self.wait
                record_status = 'fallback'
            else:
                self.latest_x_fsp = self.opt_model.x_fsp_value.copy()
                self.latest_x_vsp = self.opt_model.x_vsp_value.copy()
                self._fallback_offset = self.wait
                x_fsp = self.opt_model.x_fsp_value[:self.wait, :]
                x_vsp = self.opt_model.x_vsp_value[:self.wait, :]
                record_status = status

            realization = self.realizations.iloc[step_iloc: step_iloc + self.wait]

            # Optimizer-predicted tank volumes (always computed)
            opt_tank_volumes = {}
            for i, (tank_id, tank_data) in enumerate(self.net.tanks.iterrows()):
                vol = self.net.get_tank_vol(t=self.wait, tank_id=tank_id,
                                            x_fsp=x_fsp, x_vsp=x_vsp,
                                            sample=realization.iloc[:self.wait, 1 + i].values)
                opt_tank_volumes[tank_id] = vol

            # EPANET true volumes (when simulator present); also updates init_vol for next step
            if self.simulator is not None:
                x_fsp_df = pd.DataFrame(x_fsp, columns=self.net.fsp.index)
                epanet_volumes, epanet_energy = self.simulator.step(x_fsp_df, self.wait)
                for tank_id in self.net.tanks.index:
                    self.net.tanks.loc[tank_id, 'init_vol'] = epanet_volumes.iloc[-1][tank_id]
            else:
                epanet_volumes = None
                epanet_energy = None
                for tank_id in self.net.tanks.index:
                    self.net.tanks.loc[tank_id, 'init_vol'] = opt_tank_volumes[tank_id][-1]

            df = self._record_step(realization, x_fsp, x_vsp, opt_tank_volumes, epanet_volumes, epanet_energy, record_status, solver_time, solver_mip_gap)
            self.results = pd.concat([self.results, df], ignore_index=True)
            self.t_idx += 1

            fsp_power = self.net.fsp['power'].values
            step_power = x_fsp @ fsp_power  # (wait,) total kW per implemented step
            pred_cost = (step_power * self.last_forecast[self.prices_col].values[:self.wait] / 1000).sum()
            if self.simulator is not None:
                actual_cost = df['epanet_cost'].sum()
            else:
                actual_cost = (step_power * realization[self.prices_col].values[:self.wait] / 1000).sum()
            bar.set_postfix_str(
                f"{self.current_time.strftime('%m-%d %H:%M')} | {status} | "
                f"pred: {pred_cost:.1f} | actual: {actual_cost:.1f} | infeasible: {self.n_infeasible}"
            )

        if self.simulator is not None:
            self.simulator.close()

    def _compute_violations(self, tank_volumes_dict):
        n_min_viol = np.zeros(self.wait, dtype=int)
        min_viol_vol = np.zeros(self.wait)
        n_max_viol = np.zeros(self.wait, dtype=int)
        max_viol_vol = np.zeros(self.wait)
        for tank_id, tank_data in self.net.tanks.iterrows():
            vol = tank_volumes_dict[tank_id]
            vol = vol.values if hasattr(vol, 'values') else vol.flatten()
            n_min_viol += (vol < tank_data['min_vol']).astype(int)
            min_viol_vol += np.maximum(0.0, tank_data['min_vol'] - vol)
            n_max_viol += (vol > tank_data['max_vol']).astype(int)
            max_viol_vol += np.maximum(0.0, vol - tank_data['max_vol'])
        return n_min_viol, min_viol_vol, n_max_viol, max_viol_vol

    def _record_step(self, realization, x_fsp, x_vsp, opt_tank_volumes, epanet_volumes, epanet_energy, opt_status, solver_time, solver_mip_gap):
        opt_n_min_viol, opt_min_viol_vol, opt_n_max_viol, opt_max_viol_vol = \
            self._compute_violations(opt_tank_volumes)
        if epanet_volumes is not None:
            epanet_n_min_viol, epanet_min_viol_vol, epanet_n_max_viol, epanet_max_viol_vol = \
                self._compute_violations({tid: epanet_volumes[tid] for tid in self.net.tanks.index})
        else:
            epanet_n_min_viol = epanet_min_viol_vol = epanet_n_max_viol = epanet_max_viol_vol = None
        return pd.DataFrame({
            "idx": np.repeat(self.t_idx, self.wait),
            "time": realization.index,
            "opt_status": np.repeat(opt_status, self.wait),
            "solver_time": np.repeat(solver_time, self.wait),
            "solver_mip_gap": np.repeat(solver_mip_gap, self.wait),
            **{f"x_fsp_{i}": x_fsp[:, i] for i in range(self.net.n_fsp)},
            **{f"x_vsp_{i}": x_vsp[:, i] for i in range(self.net.n_vsp)},
            "predicted_tariff": self.last_forecast.loc[realization.index, self.prices_col].values[:self.wait],
            "realized_tariff": realization[self.prices_col].values[:self.wait],
            **{f"predicted_demand_{tank_id}":
                   self.last_forecast.loc[realization.index, tank_data["demand"]].values[:self.wait]
               for tank_id, tank_data in self.net.tanks.iterrows()},
            **{f"realized_demand_{tank_id}": realization.loc[:, tank_data["demand"]].values[:self.wait]
               for tank_id, tank_data in self.net.tanks.iterrows()},
            **{f"opt_vol_tank_{tank_id}": opt_tank_volumes[tank_id].flatten()
               for tank_id in self.net.tanks.index},
            **{f"opt_energy_{combo}": x_fsp[:, i] * self.net.fsp.loc[combo, 'power']
               for i, combo in enumerate(self.net.fsp.index)},
            "opt_n_min_violations": opt_n_min_viol,
            "opt_min_violated_vol": opt_min_viol_vol,
            "opt_n_max_violations": opt_n_max_viol,
            "opt_max_violated_vol": opt_max_viol_vol,
            **(
                {f"epanet_vol_tank_{tank_id}": epanet_volumes[tank_id].values
                 for tank_id in self.net.tanks.index}
                if epanet_volumes is not None else {}
            ),
            **(
                {f"epanet_energy_{pump_id}": epanet_energy[pump_id].values
                 for pump_id in epanet_energy.columns}
                if epanet_energy is not None else {}
            ),
            **(
                {"epanet_cost": (epanet_energy.values *
                                 realization[self.prices_col].values[:self.wait, np.newaxis] / 1000).sum(axis=1)}
                if epanet_energy is not None else {}
            ),
            **(
                {
                    "epanet_n_min_violations": epanet_n_min_viol,
                    "epanet_min_violated_vol": epanet_min_viol_vol,
                    "epanet_n_max_violations": epanet_n_max_viol,
                    "epanet_max_violated_vol": epanet_max_viol_vol,
                }
                if epanet_n_min_viol is not None else {}
            ),
        })

    def plot_results(self):
        fig, axes = plt.subplots(nrows=self.net.n_tanks + 1, sharex=True)
        for i, tank_id in enumerate(self.net.tanks.index):
            epanet_col = f"epanet_vol_tank_{tank_id}"
            vol_col = epanet_col if epanet_col in self.results.columns else f"opt_vol_tank_{tank_id}"
            axes[i].plot(self.results[vol_col], label=vol_col)
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
