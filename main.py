import argparse
import shutil
import time
from pathlib import Path
import cvxpy as cp
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
from opt import BaseOptModel, RO, SP, Simulation, extract_problem_stats
from mpc import MPC
from epanet_simulator import EPANETSimulator

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
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

    panels = []
    for col in demand_cols:
        panels.append({
            'time': actuals_aligned.index,
            'predicted': forecast_aligned[col].values,
            'realized': actuals_aligned[col].values,
            'ylabel': 'Demand\n($m^3/h$)',
        })
    panels.append({
        'time': actuals_aligned.index,
        'predicted': forecast_aligned[prices_col].values,
        'realized': actuals_aligned[prices_col].values,
        'ylabel': 'Tariff\n(€/MWh)',
    })

    return graphs.plot_predicted_vs_realized(panels)


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
    t0 = time.time()
    model.formulate()
    status, cost = model.solve()
    total_time = time.time() - t0

    if status not in cp.settings.SOLUTION_PRESENT:
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
        'inaccurate': status != 'optimal',
        'solver_time': model.solve_time,
        'total_time': total_time,
        'mip_gap': model.solver_mip_gap,
    }

    if nominal_cost is not None:
        result['por'] = avg_cost / (nominal_cost / 1000)

    return result


def por(uset_data, forecast, methods, net_path, t=24, n=500, milp=False, plot=True,
        sampling_method='normal', seed=42, omega_to_alpha='gaussian', stat_analyze=True,
        final_vol_constraint=True, solver_params=None, n_binary_steps=None):
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
        'final_vol_constraint': final_vol_constraint,
        'solver_params': solver_params,
        'n_binary_steps': n_binary_steps,
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
        psa.plot_por_results(df, sp_theoretical_data=sp_theoretical_data,
                             omega_to_alpha=omega_to_alpha, stat_analyze=stat_analyze)

    return df


def run_static_optimization(model_class, forecast, uset_estimator, net_path='data/price_ostfeld',
                            t=24, n_sim=500, milp=True, plot=True, final_vol_constraint=True,
                            solver_params=None, n_binary_steps=None, **model_params):
    """
    Run a single static optimization with optional plotting.
    """
    net = LPNetwork(net_path)
    params = {'network': net, 't': t, 'forecast': forecast, 'uset_estimator': uset_estimator, 'milp': milp,
              'final_vol_constraint': final_vol_constraint, 'solver_params': solver_params,
              'n_binary_steps': n_binary_steps, **model_params}

    model = model_class(**params)
    model.formulate()
    status, cost = model.solve()

    if status not in cp.settings.SOLUTION_PRESENT:
        return None

    sim = Simulation(model, sampling_method=model_params.get('sampling_method', 'normal'), n=n_sim)
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


def run_mpc_experiment(dm, uset_estimator, realizations, prices_col, demand_cols, opt_method,
                       start_datetime, step_horizon, wait, experiment_horizon,
                       forecast_method, net_path, milp, plot, holiday_country, final_vol_constraint,
                       n_binary_steps=None, forecast_params=None, simulator=None, export_inp_path=None, **opt_params):
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
        holiday_country: ISO country code for holidays
        **opt_params: Additional optimization parameters (omega, gamma, alpha, etc.)
    """
    mpc = MPC(net_path=net_path, opt_method=opt_method, db=dm, uset_estimator=uset_estimator,
              start_datetime=start_datetime, step_horizon=step_horizon, wait=wait,
              experiment_horizon=experiment_horizon, realizations=realizations,
              prices_col=prices_col, demand_cols=demand_cols,
              forecast_method=forecast_method, forecast_params=forecast_params,
              milp=milp, holiday_country=holiday_country, simulator=simulator,
              export_inp_path=export_inp_path,
              final_vol_constraint=final_vol_constraint, n_binary_steps=n_binary_steps,
              **opt_params)
    mpc.init_opt_model()
    mpc.run()

    if plot:
        mpc.plot_results()

    return mpc


def _compute_metrics_from_df(results, net, n_infeasible=None):
    """
    Compute MPC performance metrics from a trajectory DataFrame.

    n_infeasible: if None, approximated from opt_status == 'fallback' rows.
    Steps that failed with no fallback are skipped in the MPC loop and never
    recorded, so the approximation may undercount by that amount.
    """
    if results.empty:
        return {
            'actual_cost': float('nan'), 'predicted_cost': float('nan'),
            'n_min_violations': 0, 'min_violated_vol': 0.0,
            'n_max_violations': 0, 'max_violated_vol': 0.0,
            'violated_vol': 0.0, 'constraint_satisfaction_rate': float('nan'),
            'price_mae': float('nan'), 'demand_mae': float('nan'),
            'n_infeasible': n_infeasible if n_infeasible is not None else 0,
        }

    if n_infeasible is None:
        n_infeasible = int((results['opt_status'] == 'fallback').sum())

    n_steps = len(results)

    fsp_powers = net.fsp['power'].values
    fsp_cols = [f'x_fsp_{i}' for i in range(net.n_fsp)]
    total_fsp_power = results[fsp_cols].values @ fsp_powers
    if 'epanet_cost' in results.columns:
        actual_cost = float(results['epanet_cost'].sum())
    else:
        actual_cost = float((results['realized_tariff'].values * total_fsp_power).sum()) / 1000
    predicted_cost = float((results['predicted_tariff'].values * total_fsp_power).sum()) / 1000

    prefix = 'epanet' if 'epanet_n_min_violations' in results.columns else 'opt'
    n_min_violations = int(results[f'{prefix}_n_min_violations'].sum())
    min_violated_vol = float(results[f'{prefix}_min_violated_vol'].sum())
    n_max_violations = int(results[f'{prefix}_n_max_violations'].sum())
    max_violated_vol = float(results[f'{prefix}_max_violated_vol'].sum())
    n_violations = n_min_violations + n_max_violations
    constraint_satisfaction_rate = 1.0 - n_violations / (n_steps * net.n_tanks)

    price_mae = float(np.mean(np.abs(results['predicted_tariff'].values - results['realized_tariff'].values)))
    demand_maes = []
    for tank_id in net.tanks.index:
        demand_maes.append(np.mean(np.abs(
            results[f'predicted_demand_{tank_id}'].values - results[f'realized_demand_{tank_id}'].values
        )))
    demand_mae = float(np.mean(demand_maes))

    mip_gap_col = results['solver_mip_gap'] if 'solver_mip_gap' in results.columns else pd.Series(dtype=float)
    mean_mip_gap = float(np.nanmean(mip_gap_col)) if not mip_gap_col.empty else float('nan')
    max_mip_gap = float(np.nanmax(mip_gap_col)) if not mip_gap_col.empty else float('nan')

    solver_time_col = results['solver_time'] if 'solver_time' in results.columns else pd.Series(dtype=float)
    mean_solver_time = float(np.nanmean(solver_time_col)) if not solver_time_col.empty else float('nan')
    std_solver_time = float(np.nanstd(solver_time_col)) if not solver_time_col.empty else float('nan')

    step_time_col = results['step_total_time'] if 'step_total_time' in results.columns else pd.Series(dtype=float)
    mean_step_total_time = float(np.nanmean(step_time_col)) if not step_time_col.empty else float('nan')
    std_step_total_time = float(np.nanstd(step_time_col)) if not step_time_col.empty else float('nan')

    return {
        'actual_cost':                  actual_cost,
        'predicted_cost':               predicted_cost,
        'n_min_violations':             n_min_violations,
        'min_violated_vol':             min_violated_vol,
        'n_max_violations':             n_max_violations,
        'max_violated_vol':             max_violated_vol,
        'violated_vol':                 min_violated_vol + max_violated_vol,
        'constraint_satisfaction_rate': constraint_satisfaction_rate,
        'price_mae':                    price_mae,
        'demand_mae':                   demand_mae,
        'n_infeasible':                 n_infeasible,
        'mean_mip_gap':                 mean_mip_gap,
        'max_mip_gap':                  max_mip_gap,
        'mean_solver_time':             mean_solver_time,
        'std_solver_time':              std_solver_time,
        'mean_step_total_time':         mean_step_total_time,
        'std_step_total_time':          std_step_total_time,
    }


def _compute_mpc_metrics(mpc):
    results = mpc.results
    net = mpc.net

    if results.empty:
        total_steps = mpc.experiment_horizon // mpc.wait
        print(f"  Warning: MPC produced no recorded steps ({mpc.n_infeasible} infeasible out of {total_steps})")
        return {
            'actual_cost': float('nan'), 'predicted_cost': float('nan'),
            'n_min_violations': 0, 'min_violated_vol': 0.0,
            'n_max_violations': 0, 'max_violated_vol': 0.0,
            'violated_vol': 0.0, 'constraint_satisfaction_rate': float('nan'),
            'price_mae': float('nan'), 'demand_mae': float('nan'),
            'n_infeasible': mpc.n_infeasible,
        }

    return _compute_metrics_from_df(results, net, n_infeasible=mpc.n_infeasible)


def _merge_kwargs(common_kwargs, method_params):
    """Merge common and per-method kwargs with method params taking priority.

    solver_params is deep-merged so options-level defaults (e.g. TimeLimit) are
    not silently dropped when the method sweep overrides Threads or Seed.
    """
    merged = {**common_kwargs, **method_params}
    if 'solver_params' in common_kwargs and 'solver_params' in method_params:
        merged['solver_params'] = {
            **(common_kwargs['solver_params'] or {}),
            **(method_params['solver_params'] or {}),
        }
    return merged


def _run_method_worker(worker_args):
    """Module-level worker for parallel MPC sweep. Must be at module level for Windows multiprocessing."""
    method = worker_args['method']
    common_kwargs = worker_args['common_kwargs']
    sim_cfg = worker_args['sim_cfg']
    network_path = worker_args['network_path']
    prices_col = worker_args['prices_col']
    demand_cols = worker_args['demand_cols']
    final_vol_constraint = worker_args['final_vol_constraint']

    label = method['label']
    opt_method = method['opt_method']
    params = method.get('params', {})
    group = method.get('group', label)
    export_inp_path = worker_args.get('export_inp_path')

    simulator = None
    if sim_cfg:
        sim_net = LPNetwork(network_path)
        simulator = EPANETSimulator(
            network=sim_net,
            inp_file=sim_cfg['inp_file'],
            prices_col=prices_col,
            demand_cols=demand_cols,
        )

    print(f"  [{label}] running...")
    mpc = run_mpc_experiment(opt_method=opt_method, simulator=simulator,
                             final_vol_constraint=final_vol_constraint,
                             export_inp_path=export_inp_path,
                             **_merge_kwargs(common_kwargs, params))
    mpc_metrics = _compute_mpc_metrics(mpc)
    problem_stats = extract_problem_stats(mpc.opt_model)

    traj = mpc.results.copy()
    traj['model'] = opt_method
    traj['label'] = label
    traj['group'] = group

    return {'label': label, 'group': group, 'opt_method': opt_method,
            'params': params, 'metrics': mpc_metrics, 'trajectory': traj,
            'problem_stats': problem_stats}


def _label_to_filename(label):
    safe = re.sub(r'[^\w.]', '_', label)
    return safe[:60]


def consolidate_sweep_output(output_dir):
    """
    Merge per-method traj_*.csv / sum_*.csv files into summary.csv + trajectories.csv,
    then delete the per-method files.

    Useful when a sweep was interrupted and the final consolidated files were not written,
    or to manually clean up a directory that still has per-method checkpoint files.

    Returns (summary_df, trajectories_df), or (None, None) if no per-method files exist.
    """
    output_dir = Path(output_dir)
    sum_files = sorted(output_dir.glob('sum_*.csv'))
    traj_files = sorted(output_dir.glob('traj_*.csv'))

    if not sum_files:
        print(f"  No per-method files found in {output_dir}")
        return None, None

    summary = pd.concat([pd.read_csv(f) for f in sum_files], ignore_index=True)
    trajectories = pd.concat([pd.read_csv(f) for f in traj_files], ignore_index=True) if traj_files else pd.DataFrame()

    summary.to_csv(output_dir / 'summary.csv', index=False)
    trajectories.to_csv(output_dir / 'trajectories.csv', index=False)

    for f in sum_files + traj_files:
        f.unlink()

    print(f"  Consolidated {len(sum_files)} method files -> summary.csv + trajectories.csv, deleted per-method files")
    return summary, trajectories


def _run_mpc_sweep_from_config(config, dm, uset_estimator, config_path=None):
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
        step_horizon=mpc_cfg.step_horizon,
        wait=mpc_cfg.wait,
        experiment_horizon=exp.experiment_period.horizon,
        forecast_method=config.forecast.model,
        forecast_params=config.forecast.get('params', {}),
        net_path=config.data.network_path,
        milp=opts['milp'],
        holiday_country=config.forecast.holiday_country,
        plot=False,
        solver_params=opts.get('solver_params'),
        n_binary_steps=opts.get('n_binary_steps'),
    )

    output_dir = opts.get('output_dir', None)
    plot_names = opts.get('plots', [])

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if config_path is not None:
            shutil.copy2(config_path, f"{output_dir}/{Path(config_path).name}")

    sim_cfg = opts.get('simulator')
    n_workers = opts.get('n_workers', 1)

    methods = cfg.expand_methods(exp.methods)
    rows = []
    trajectory_parts = []
    problem_stats_rows = []

    if n_workers > 1:
        futures_to_method = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for method in methods:
                label_safe = _label_to_filename(method['label'])
                export_inp_path = None
                if opts.get('export_inp') and output_dir:
                    export_inp_path = str(Path(output_dir) / f"sched_{label_safe}.inp")
                futures_to_method[executor.submit(_run_method_worker, {
                        'method': method,
                        'common_kwargs': common_kwargs,
                        'sim_cfg': sim_cfg,
                        'network_path': config.data.network_path,
                        'prices_col': config.columns.prices_col,
                        'demand_cols': config.columns.demand_cols,
                        'final_vol_constraint': opts.get('final_vol_constraint', True),
                        'export_inp_path': export_inp_path,
                    })] = method

            for future in as_completed(futures_to_method):
                result = future.result()
                label = result['label']
                label_safe = _label_to_filename(label)
                row = {'label': label, 'group': result['group'], 'opt_method': result['opt_method'],
                       **result['params'], **result['metrics']}
                rows.append(row)
                problem_stats_rows.append({'label': label, 'group': result['group'],
                                           'opt_method': result['opt_method'], **result['problem_stats']})
                traj = result['trajectory']
                trajectory_parts.append(traj)
                if output_dir is not None:
                    traj.to_csv(Path(output_dir) / f"traj_{label_safe}.csv", index=False)
                    pd.DataFrame([row]).to_csv(Path(output_dir) / f"sum_{label_safe}.csv", index=False)
                    pd.DataFrame(rows).to_csv(f"{output_dir}/summary.csv", index=False)
                    print(f"  [{label}] saved to: {output_dir}/")
    else:
        for method in methods:
            label = method['label']
            opt_method = method['opt_method']
            params = method.get('params', {})
            group = method.get('group', label)
            label_safe = _label_to_filename(label)

            simulator = None
            if sim_cfg:
                sim_net = LPNetwork(config.data.network_path)
                simulator = EPANETSimulator(
                    network=sim_net,
                    inp_file=sim_cfg['inp_file'],
                    prices_col=config.columns.prices_col,
                    demand_cols=config.columns.demand_cols,
                )

            export_inp_path = None
            if opts.get('export_inp') and output_dir:
                export_inp_path = str(Path(output_dir) / f"sched_{label_safe}.inp")

            print(f"  [{label}] running...")
            mpc = run_mpc_experiment(opt_method=opt_method, simulator=simulator,
                                     final_vol_constraint=opts.get('final_vol_constraint', True),
                                     export_inp_path=export_inp_path,
                                     **_merge_kwargs(common_kwargs, params))
            mpc_metrics = _compute_mpc_metrics(mpc)
            row = {'label': label, 'group': group, 'opt_method': opt_method, **params, **mpc_metrics}
            rows.append(row)
            problem_stats_rows.append({'label': label, 'group': group, 'opt_method': opt_method,
                                       **extract_problem_stats(mpc.opt_model)})

            traj = mpc.results.copy()
            traj['model'] = opt_method
            traj['label'] = label
            traj['group'] = group
            trajectory_parts.append(traj)

            if output_dir is not None:
                traj.to_csv(Path(output_dir) / f"traj_{label_safe}.csv", index=False)
                pd.DataFrame([row]).to_csv(Path(output_dir) / f"sum_{label_safe}.csv", index=False)
                pd.DataFrame(rows).to_csv(f"{output_dir}/summary.csv", index=False)
                print(f"  [{label}] saved to: {output_dir}/")

    results = pd.DataFrame(rows)
    all_traj = pd.concat(trajectory_parts, ignore_index=True) if trajectory_parts else pd.DataFrame()

    if output_dir is not None:
        pd.DataFrame(rows).to_csv(f"{output_dir}/summary.csv", index=False)
        all_traj.to_csv(f"{output_dir}/trajectories.csv", index=False)
        pd.DataFrame(problem_stats_rows).to_csv(f"{output_dir}/problem_stats.csv", index=False)
        for f in list(Path(output_dir).glob('sum_*.csv')) + list(Path(output_dir).glob('traj_*.csv')):
            f.unlink()
        print(f"  All methods complete. Outputs in: {output_dir}/")

        net = LPNetwork(config.data.network_path)
        g = graphs.MPCGraphs(all_traj, net)
        plot_fns = {
            'tank_trajectories': lambda: g.plot_tank_trajectories(),
            'hydraulic_accuracy': lambda: g.plot_hydraulic_accuracy(),
            'hydraulic_scatter': lambda: g.plot_hydraulic_scatter(),
            'energy_accuracy': lambda: g.plot_energy_accuracy(),
            'por': lambda: g.plot_por(results),
            'forecast_accuracy': lambda: graphs.plot_mpc_forecast_accuracy(all_traj),
            'cumulative_metrics': lambda: g.plot_cumulative_metrics(),
            'cumulative_energy_by_price': lambda: g.plot_cumulative_energy_by_price(),
        }
        for name in plot_names:
            fig_out = plot_fns[name]()
            if isinstance(fig_out, tuple):
                fig_out = fig_out[0]
            fig_out.savefig(f"{output_dir}/{name}.png", dpi=150, bbox_inches='tight')
            plt.close(fig_out)
            print(f"  Saved: {output_dir}/{name}.png")

    return results


def run_mpc_sweep(config_path):
    config = cfg.load_config(config_path)

    dm = data_manager.DataManager(config.data.demands_path, config.data.tariffs_path,
                                  prices_col=config.columns.prices_col)
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

    return _run_mpc_sweep_from_config(config, dm, uset_estimator, config_path=config_path)


def run_from_config(config_path):
    config = cfg.load_config(config_path)
    exp = config.experiment

    # Setup DataManager
    dm = data_manager.DataManager(config.data.demands_path, config.data.tariffs_path,
                                  prices_col=config.columns.prices_col)
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
        return _run_mpc_sweep_from_config(config, dm, uset_estimator, config_path=config_path)

    # For static/POR: set up a standalone forecaster for the fixed forecast period
    forecast_start = pd.to_datetime(config.experiment.forecast_period.start)
    last_midnight = forecast_start.normalize() - pd.Timedelta(hours=1)
    if dm.data.index.tz is not None:
        last_midnight = last_midnight.tz_localize(dm.data.index.tz)
    buffer = forecast.RealizationsBuffer(dm.data.loc[:last_midnight])

    forecast_model = config.forecast.model
    forecast_params = config.forecast.get('params', {})
    forecaster = train_forecaster(
        training_data=dm.forecast_training_data,
        prices_col=config.columns.prices_col,
        demand_cols=config.columns.demand_cols,
        model_name=forecast_model,
        holiday_country=config.forecast.holiday_country,
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
    model_class = cfg.get_model_class(exp.model.opt_method)
    opts = exp.get('options', {})

    return run_static_optimization(
        model_class=model_class,
        forecast=forecast_df,
        uset_estimator=uset_estimator,
        net_path=config.data.network_path,
        t=opts['t'],
        n_sim=exp['simulation']['n_samples'],
        milp=opts['milp'],
        plot=opts['plot'],
        final_vol_constraint=opts.get('final_vol_constraint', True),
        solver_params=opts.get('solver_params'),
        n_binary_steps=opts.get('n_binary_steps'),
        **exp.model.get('params', {})
    )


def _run_por_from_config(config, dm, forecaster, uset_estimator):
    """Run POR comparison from config."""
    exp = config.experiment
    forecast_df = forecaster.generate_forecast(
        exp.forecast_period.start,
        exp.forecast_period.end
    )

    methods = cfg.expand_methods(exp.methods)
    opts = exp.get('options', {})
    return por(
        uset_data=dm.uncertainty_training_data,
        forecast=forecast_df,
        methods=methods,
        net_path=config.data.network_path,
        t=opts['t'],
        n=exp['simulation']['n_samples'],
        milp=opts['milp'],
        plot=opts['plot'],
        sampling_method=opts.get('sampling_method', 'normal'),
        omega_to_alpha=opts.get('omega_to_alpha', 'gaussian'),
        stat_analyze=opts.get('stat_analyze', True),
        final_vol_constraint=opts.get('final_vol_constraint', True),
        solver_params=opts.get('solver_params'),
        n_binary_steps=opts.get('n_binary_steps'),
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
        )

    return run_mpc_experiment(
        dm=dm,
        uset_estimator=uset_estimator,
        realizations=dm.data,
        prices_col=config.columns.prices_col,
        demand_cols=config.columns.demand_cols,
        opt_method=mpc_cfg.opt_method,
        start_datetime=pd.to_datetime(exp.experiment_period.start),
        step_horizon=mpc_cfg.step_horizon,
        wait=mpc_cfg.wait,
        experiment_horizon=exp.experiment_period.horizon,
        forecast_method=config.forecast.model,
        forecast_params=config.forecast.get('params', {}),
        net_path=config.data.network_path,
        milp=opts['milp'],
        plot=opts['plot'],
        holiday_country=config.forecast.holiday_country,
        simulator=simulator,
        final_vol_constraint=opts.get('final_vol_constraint', True),
        solver_params=opts.get('solver_params'),
        n_binary_steps=opts.get('n_binary_steps'),
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
