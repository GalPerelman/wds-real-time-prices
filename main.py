import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import data_manager
import forecast
import graphs
import uncertainty_utils
import config as cfg
import por_statistical_analysis as psa
from network import LPNetwork
from opt import BaseOptModel, RO, SP, Simulation
from mpc import MPC
from epanet_simulator import EPANETSimulator

from typing import Dict, List, Any, Optional
from itertools import product


def train_forecaster(training_data, prices_col, demand_cols, model_name='xgb', holiday_country='HU',
                     buffer=None, random_state=42, **model_params):
    """
    Train a forecaster on the provided training data.

    Args:
        training_data: DataFrame with datetime index containing prices and demands
        prices_col: Column name for electricity prices (e.g., 'tariff')
        demand_cols: List of column names for demands (e.g., ['a', 'b', 'c'])
        model_name: Forecasting model ('xgb', 'rf', 'naive')
        holiday_country: ISO country code for holidays (default 'HU')
        buffer: RealizationsBuffer instance (required for naive method)
        random_state: Random seed for ML models for reproducibility (default 42)
        **model_params: Model-specific hyperparameters, e.g.:
            - naive: n_weeks=4
            - xgb/rf: n_estimators=100, etc.

    Returns:
        Fitted Forecast object ready for prediction
    """
    f = forecast.Forecast(training_data=training_data, prices_col=prices_col, demand_cols=demand_cols,
                          holiday_country=holiday_country, buffer=buffer, random_state=random_state)
    f.preprocess()
    f.fit(model_name=model_name, target="demand", **model_params)
    f.fit(model_name=model_name, target="price", **model_params)
    return f


def plot_forecast_comparison(forecast_df, actuals_df, demand_cols, prices_col):
    common_index = forecast_df.index.intersection(actuals_df.index)
    forecast_aligned = forecast_df.loc[common_index]
    actuals_aligned = actuals_df.loc[common_index]

    n_cols = len(forecast_df.columns)
    fig, axes = plt.subplots(nrows=n_cols, ncols=1, sharex=True, figsize=(6, 5))
    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(demand_cols):
        axes[i].plot(actuals_aligned.index, actuals_aligned[col], label='Actual', color='C0')
        axes[i].plot(forecast_aligned.index, forecast_aligned[col], label='Forecast', color='C1')
        axes[i].set_ylabel(f'Demand\n($m^3/h$)')
        if i == 0:
            axes[i].legend(loc='upper right')

        # Calculate and display metrics
        mae = np.mean(np.abs(actuals_aligned[col].values - forecast_aligned[col].values))
        axes[i].text(0.02, 0.82, f'MAE={mae:.2f}', transform=axes[i].transAxes, horizontalalignment='left')

    axes[-1].plot(actuals_aligned.index, actuals_aligned[prices_col], label='Actual', color='C0')
    axes[-1].plot(forecast_aligned.index, forecast_aligned[prices_col], label='Forecast', color='C1')
    mae = np.mean(np.abs(actuals_aligned[prices_col].values - forecast_aligned[prices_col].values))
    axes[-1].text(0.02, 0.82, f'MAE={mae:.2f}', transform=axes[-1].transAxes, horizontalalignment='left')
    axes[-1].set_ylabel('Tariff\n(€/MWh)')
    axes[-1].set_xlabel('Time')

    fig.align_ylabels()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.08)
    fig.autofmt_xdate()
    return fig, axes


def run_single_model(model_class, model_params, sim_n, nominal_cost=None, sample=None):
    """
    Run a single optimization model and return metrics.

    Args:
        model_class: Optimization model class (BaseOptModel, RO, SP)
        model_params: Dict of parameters to pass to model constructor
        sim_n: Number of simulation samples (ignored if sample is provided)
        nominal_cost: Optional nominal LP cost for POR calculation
        sample: Optional pre-generated sample array for consistent evaluation

    Returns dict with: objective, avg_cost, reliability, violated_vol, por (if nominal_cost provided)
    Returns None if optimization fails.
    """
    # For SP models, pass the shared sample for SAA objective
    if model_class == SP and sample is not None:
        model_params = {**model_params, 'sample': sample}

    model = model_class(**model_params)
    model.formulate()
    status, cost = model.solve()

    if status != 'optimal':
        return None

    # Use shared sample for simulation evaluation
    sim = Simulation(model, sampling_method="normal", n=sim_n, sample=sample)
    violation_rate = sim.summarize()
    avg_cost = sim.costs.mean() / 1000

    result = {
        'objective': cost / 1000,
        'avg_cost': avg_cost,
        'reliability': 1 - violation_rate,
        'violated_vol': sim.violation_volume.sum().sum(),
    }

    if nominal_cost is not None:
        result['por'] = avg_cost / (nominal_cost / 1000)

    return result


def por(uset_data, forecast, methods, net_path, group_styles=None, t=24, n=500, milp=False, plot=True,
        sampling_method='normal', seed=42, omega_to_alpha='gaussian', stat_analyze=True):
    """
    Price of Robustness (POR) comparison across multiple optimization methods.
    """
    net = LPNetwork(net_path)

    # Fit uncertainty estimator once (reused by all models)
    uset_estimator = uncertainty_utils.SpatioTemporalEllipsoidUncertaintyEstimator(
        seasonal_demean=True,
        drop_incomplete_windows=True
    )
    uset_estimator.fit(uset_data)

    # Common params for all models
    common_params = {
        'network': net,
        't': t,
        'forecast': forecast,
        'uset_estimator': uset_estimator,
        'milp': milp,
    }

    # Get nominal cost from LP for POR calculation
    # Also use this to get mu and sigma for generating shared sample
    lp = BaseOptModel(**common_params)
    lp.formulate()
    _, nominal_cost = lp.solve()

    # Generate shared sample ONCE for all experiments
    # This ensures all methods are optimized (SP) and evaluated against identical scenarios
    start_hour = forecast.index[0].hour
    lp.get_ellipsoid_uncertainty_set(start_hour=start_hour)
    mu = lp.get_nominal()
    cov = lp.sigma

    shared_sample = uncertainty_utils.Sampler.generate_shared_sample(
        mu=mu, cov=cov, n=n, sampling_method=sampling_method, seed=seed, use_stable_sampling=True)

    # Run all methods with shared sample
    results = []
    for method in methods:
        label = method['label']
        group = method.get('group', label)
        model_class = cfg.get_model_class(method['opt_method'])
        model_params = {**common_params, **method.get('params', {})}
        result = run_single_model(model_class, model_params, sim_n=n, nominal_cost=nominal_cost, sample=shared_sample)

        if result is not None:
            result['label'] = label
            result['group'] = group
            # Store method-specific params for reference
            for k, v in method.get('params', {}).items():
                result[k] = v
            results.append(result)
        else:
            print(f"Warning: {label} ({group}) did not converge to optimal")

    df = pd.DataFrame(results)

    sp_theoretical_data = None
    if stat_analyze:
        sp_methods = [m for m in methods if cfg.get_model_class(m['opt_method']) == SP]
        if sp_methods:
            first_sp_params = {**common_params, **sp_methods[0].get('params', {}), 'sample': shared_sample}
            sp_ref = SP(**first_sp_params)
            sp_ref.formulate()
            sp_theoretical_data = {'shared_sample': shared_sample, 'mu': mu,
                                   'delta': sp_ref.delta, 'sigma_rhs': sp_ref.sigma_rhs, 't': t}

    if plot and len(df) > 0:
        psa.plot_por_results(df, group_styles, sp_theoretical_data=sp_theoretical_data,
                             omega_to_alpha=omega_to_alpha, stat_analyze=stat_analyze)

    return df


def run_static_optimization(model_class, forecast, uset_estimator, net_path='data/multi_tank',
                            t=24, n_sim=500, milp=True, plot=True, **model_params):
    """
    Run a single static optimization with optional plotting.
    """
    net = LPNetwork(net_path)
    params = {'network': net, 't': t, 'forecast': forecast, 'uset_estimator': uset_estimator, 'milp': milp,
              **model_params}

    model = model_class(**params)
    model.formulate()
    status, cost = model.solve()

    if status != 'optimal':
        return None

    sim = Simulation(model, sampling_method="normal", n=n_sim)
    violation_rate = sim.summarize()

    result = {
        'objective': cost / 1000,
        'avg_cost': sim.costs.mean() / 1000,
        'reliability': 1 - violation_rate,
        'violated_vol': sim.violation_volume.sum().sum(),
    }

    if plot:
        graphs.OptGraphs(model).all_tanks(plot_tariff=True, ncols=1, sample=sim.sample)

    return result


def run_mpc_experiment(dm, uset_estimator, realizations, prices_col, demand_cols, opt_method='ro',
                       start_datetime=None, step_horizon=24, wait=1, experiment_horizon=48,
                       forecast_method='xgb', forecast_params=None, net_path='data/multi_tank',
                       milp=True, plot=True, holiday_country='HU', simulator=None, **opt_params):
    """
    Run an MPC rolling horizon experiment.

    Args:
        dm: DataManager with forecast_training_data set
        uset_estimator: Pre-fitted uncertainty estimator
        realizations: DataFrame with actual values for the experiment period
        prices_col: Column name for electricity prices (e.g., 'tariff')
        demand_cols: List of column names for demands (e.g., ['a', 'b', 'c'])
        opt_method: Optimization method ('lp', 'ro', 'sp')
        start_datetime: Start of experiment
        step_horizon: Optimization horizon in hours
        wait: Hours between re-optimization
        experiment_horizon: Total experiment duration in hours
        forecast_method: Forecasting model ('xgb', 'rf')
        net_path: Path to network data
        milp: Use MILP formulation
        plot: Generate plots
        holiday_country: ISO country code for holidays (default 'HU')
        **opt_params: Additional optimization parameters (omega, gamma, alpha, etc.)
    """
    mpc = MPC(net_path=net_path, opt_method=opt_method, db=dm, uset_estimator=uset_estimator,
              start_datetime=start_datetime, step_horizon=step_horizon, wait=wait,
              experiment_horizon=experiment_horizon, realizations=realizations,
              prices_col=prices_col, demand_cols=demand_cols,
              forecast_method=forecast_method, forecast_params=forecast_params,
              milp=milp, holiday_country=holiday_country, simulator=simulator,
              **opt_params)
    mpc.init_opt_model()
    mpc.run()

    if plot:
        mpc.plot_results()

    return mpc


def _compute_mpc_metrics(mpc):
    """
    Compute performance metrics from a completed MPC run.

    Returns a dict with:
        actual_cost          - total cost at realized prices (€)
        predicted_cost       - total cost at forecast prices with same schedule (€)
        n_violations         - number of (tank, timestep) pairs with bound violations
        violated_vol         - total violated volume (m³), summed over all tanks and steps
        constraint_satisfaction_rate - fraction of (tank, timestep) pairs without violation
        price_mae            - MAE of price forecast over implemented steps (€/MWh)
        demand_mae           - MAE of demand forecast over implemented steps, averaged over tanks (m³/h)
        n_infeasible         - number of solver failures
    """
    results = mpc.results
    net = mpc.net
    n_steps = len(results)

    # Actual vs predicted cost (FSP only — VSPs have zero electricity cost per objective())
    fsp_powers = net.fsp['power'].values
    fsp_cols = [f'x_fsp_{i}' for i in range(net.n_fsp)]
    total_fsp_power = results[fsp_cols].values @ fsp_powers
    actual_cost    = float((results['realized_tariff'].values  * total_fsp_power).sum())
    predicted_cost = float((results['predicted_tariff'].values * total_fsp_power).sum())

    # Constraint violations — summed from per-step columns computed in MPC._record_step
    viol_col = 'epanet_n_violations' if 'epanet_n_violations' in results.columns else 'opt_n_violations'
    vvol_col = 'epanet_violated_vol' if 'epanet_violated_vol' in results.columns else 'opt_violated_vol'
    n_violations = int(results[viol_col].sum())
    violated_vol = float(results[vvol_col].sum())
    constraint_satisfaction_rate = 1.0 - n_violations / (n_steps * net.n_tanks)

    # Forecast MAE over implemented steps
    price_mae = float(np.mean(np.abs(results['predicted_tariff'].values - results['realized_tariff'].values)))
    demand_maes = []
    for tank_id, tank_data in net.tanks.iterrows():
        demand_maes.append(np.mean(np.abs(
            results[f'predicted_demand_{tank_id}'].values - results[f'realized_demand_{tank_id}'].values
        )))
    demand_mae = float(np.mean(demand_maes))

    return {
        'actual_cost':                 actual_cost,
        'predicted_cost':              predicted_cost,
        'n_violations':                n_violations,
        'violated_vol':                violated_vol,
        'constraint_satisfaction_rate': constraint_satisfaction_rate,
        'price_mae':                   price_mae,
        'demand_mae':                  demand_mae,
        'n_infeasible':                mpc.n_infeasible,
    }


def _run_mpc_sweep_from_config(config, dm, uset_estimator):
    exp = config.experiment
    mpc_cfg = exp.mpc
    opts = exp.get('options', {})

    common_kwargs = dict(
        dm=dm,
        uset_estimator=uset_estimator,
        realizations=dm.data,
        prices_col=config.columns.prices_col,
        demand_cols=config.columns.demand_cols,
        start_datetime=pd.to_datetime(exp.experiment_period.start),
        step_horizon=mpc_cfg.get('step_horizon', 24),
        wait=mpc_cfg.get('wait', 1),
        experiment_horizon=exp.experiment_period.horizon,
        forecast_method=config.forecast.get('model', 'xgb'),
        forecast_params=config.forecast.get('params', {}),
        net_path=config.data.network_path,
        milp=opts.get('milp', True),
        holiday_country=config.forecast.get('holiday_country', 'HU'),
        plot=False,
    )

    export_path = opts.get('export_path', None)

    sim_cfg = opts.get('simulator')

    methods = cfg.expand_methods(exp.methods)
    rows = []
    trajectory_parts = []
    for method in methods:
        label = method['label']
        opt_method = method['opt_method']
        params = method.get('params', {})

        simulator = None
        if sim_cfg:
            sim_net = LPNetwork(config.data.network_path)
            simulator = EPANETSimulator(
                network=sim_net,
                inp_file=sim_cfg['inp_file'],
                prices_col=config.columns.prices_col,
                demand_cols=config.columns.demand_cols,
                min_vol_margin_m=sim_cfg['min_vol_margin_m'],
            )

        print(f"  [{label}] running...")
        group = method.get('group', label)
        mpc = run_mpc_experiment(opt_method=opt_method, simulator=simulator, **common_kwargs, **params)
        mpc_metrics = _compute_mpc_metrics(mpc)
        rows.append({'label': label, 'group': group, 'opt_method': opt_method, **params, **mpc_metrics})

        if export_path is not None:
            traj = mpc.results.copy()
            traj['model'] = opt_method
            traj['label'] = label
            trajectory_parts.append(traj)

    if export_path is not None and trajectory_parts:
        pd.concat(trajectory_parts).to_csv(export_path + '_trajectories.csv', index=False)
        print(f"  Trajectories saved to: {export_path}")

    results = pd.DataFrame(rows)
    results.to_csv(export_path + '_summary.csv', index=False)
    return results


def run_mpc_sweep(config_path):
    config = cfg.load_config(config_path)

    dm = data_manager.DataManager(config.data.demands_path, config.data.tariffs_path)
    dm.set_forecast_training_window(
        start_date=config.training.forecast.start,
        end_date=config.training.forecast.end
    )
    dm.set_uncertainty_training_window(
        start_date=config.training.uncertainty.start,
        end_date=config.training.uncertainty.end
    )

    uset_estimator = uncertainty_utils.SpatioTemporalEllipsoidUncertaintyEstimator(
        seasonal_demean=True, drop_incomplete_windows=True
    )
    uset_estimator.fit(dm.uncertainty_training_data)

    return _run_mpc_sweep_from_config(config, dm, uset_estimator)


def run_from_config(config_path):
    config = cfg.load_config(config_path)
    exp = config.experiment

    # Setup DataManager
    dm = data_manager.DataManager(config.data.demands_path, config.data.tariffs_path)
    dm.set_forecast_training_window(
        start_date=config.training.forecast.start,
        end_date=config.training.forecast.end
    )
    dm.set_uncertainty_training_window(
        start_date=config.training.uncertainty.start,
        end_date=config.training.uncertainty.end
    )

    # Fit uncertainty estimator
    uset_estimator = uncertainty_utils.SpatioTemporalEllipsoidUncertaintyEstimator(
        seasonal_demean=True, drop_incomplete_windows=True
    )
    uset_estimator.fit(dm.uncertainty_training_data)

    # MPC manages its own internal forecaster and buffer - dispatch directly
    if exp.type == 'mpc':
        return _run_mpc_from_config(config, dm, uset_estimator)
    elif exp.type == 'mpc_sweep':
        return _run_mpc_sweep_from_config(config, dm, uset_estimator)

    # For static/POR: set up a standalone forecaster for the fixed forecast period
    forecast_start = pd.to_datetime(config.experiment.forecast_period.start)
    last_midnight = forecast_start.normalize() - pd.Timedelta(hours=1)
    if dm.data.index.tz is not None:
        last_midnight = last_midnight.tz_localize(dm.data.index.tz)
    buffer = forecast.RealizationsBuffer(dm.data.loc[:last_midnight])

    forecast_model = config.forecast.get('model', 'xgb')
    forecast_params = config.forecast.get('params', {})
    forecaster = train_forecaster(
        training_data=dm.forecast_training_data,
        prices_col=config.columns.prices_col,
        demand_cols=config.columns.demand_cols,
        model_name=forecast_model,
        holiday_country=config.forecast.get('holiday_country', 'HU'),
        buffer=buffer,
        **forecast_params
    )

    forecast_df = forecaster.generate_forecast(config.experiment.forecast_period.start,
                                               config.experiment.forecast_period.end)
    plot_forecast_comparison(forecast_df, dm.data, config.columns.demand_cols, config.columns.prices_col)

    if exp.type == 'static':
        return _run_static_from_config(config, forecaster, uset_estimator)
    elif exp.type == 'por':
        return _run_por_from_config(config, dm, forecaster, uset_estimator)
    else:
        raise ValueError(f"Unknown experiment type: {exp.type}")


def _run_static_from_config(config, forecaster, uset_estimator):
    """Run static optimization from config."""
    exp = config.experiment
    forecast_df = forecaster.generate_forecast(
        exp.forecast_period.start,
        exp.forecast_period.end
    )
    model_class = cfg.get_model_class(exp.model.get('opt_method', 'lp'))
    opts = exp.get('options', {})

    return run_static_optimization(
        model_class=model_class,
        forecast=forecast_df,
        uset_estimator=uset_estimator,
        net_path=config.data.network_path,
        t=opts.get('t', 24),
        n_sim=exp.get('simulation', {}).get('n_samples', 500),
        milp=opts.get('milp', True),
        plot=opts.get('plot', True),
        **exp.model.get('params', {})
    )


def _run_por_from_config(config, dm, forecaster, uset_estimator):
    """Run POR comparison from config."""
    exp = config.experiment
    forecast_df = forecaster.generate_forecast(
        exp.forecast_period.start,
        exp.forecast_period.end
    )

    # Expand methods with sweeps
    methods = cfg.expand_methods(exp.methods)

    # Convert group_styles if provided
    group_styles = None
    if exp.get('group_styles'):
        group_styles = {
            name: {'color': s.get('color'), 'marker': s.get('marker'), 'alpha': s.get('alpha', 1.0)}
            for name, s in exp.group_styles.items()
        }

    opts = exp.get('options', {})
    return por(
        uset_data=dm.uncertainty_training_data,
        forecast=forecast_df,
        methods=methods,
        group_styles=group_styles,
        net_path=config.data.network_path,
        t=opts.get('t', 24),
        n=exp.get('simulation', {}).get('n_samples', 500),
        milp=opts.get('milp', True),
        plot=opts.get('plot', True),
        omega_to_alpha=opts.get('omega_to_alpha', 'gaussian'),
        stat_analyze=opts.get('stat_analyze', True),
    )


def _run_mpc_from_config(config, dm, uset_estimator):
    """Run MPC experiment from config."""
    exp = config.experiment
    mpc_cfg = exp.mpc
    opts = exp.get('options', {})

    sim_cfg = opts.get('simulator')
    simulator = None
    if sim_cfg:
        sim_net = LPNetwork(config.data.network_path)
        simulator = EPANETSimulator(
            network=sim_net,
            inp_file=sim_cfg['inp_file'],
            prices_col=config.columns.prices_col,
            demand_cols=config.columns.demand_cols,
            min_vol_margin_m=sim_cfg['min_vol_margin_m'],
        )

    return run_mpc_experiment(
        dm=dm,
        uset_estimator=uset_estimator,
        realizations=dm.data,
        prices_col=config.columns.prices_col,
        demand_cols=config.columns.demand_cols,
        opt_method=mpc_cfg.get('opt_method', 'lp'),
        start_datetime=pd.to_datetime(exp.experiment_period.start),
        step_horizon=mpc_cfg.get('step_horizon', 24),
        wait=mpc_cfg.get('wait', 1),
        experiment_horizon=exp.experiment_period.horizon,
        forecast_method=config.forecast.get('model', 'xgb'),
        forecast_params=config.forecast.get('params', {}),
        net_path=config.data.network_path,
        milp=opts.get('milp', True),
        plot=opts.get('plot', True),
        holiday_country=config.forecast.get('holiday_country', 'HU'),
        **exp.get('model_params', {})
    )


if __name__ == "__main__":
    """
    python main.py --config configs/static_experiment.yaml
    """
    parser = argparse.ArgumentParser(description='WDS Optimization')
    parser.add_argument('--config', '-c', type=str, help='Path to YAML configuration file')
    args = parser.parse_args()

    print(f"Running experiment from: {args.config}")
    result = run_from_config(args.config)
    if isinstance(result, pd.DataFrame):
        print(result)
    elif isinstance(result, dict):
        for k, v in result.items():
            print(f"  {k}: {v}")

    plt.show()
