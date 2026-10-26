import datetime

import pandas as pd
import matplotlib.pyplot as plt

from network import LPNetwork
from graphs import MPCGraphs, plot_tariffs

from tariff_gradient_analysis import (
    build_gradient_table, graphical_abstract, plot_tariff_gradient_bars,
    build_forecast_table, plot_forecast_gradient_bars,
    build_case_study_table, export_case_study_latex,
)

from computational_analysis import (
    build_computational_table, to_latex_table,
    plot_binarization_sensitivity, plot_binarization_cost_bars,
)


# ---------------------------------------------------------------------------
# Tariffs overview
# ---------------------------------------------------------------------------
plot_tariffs(["data/price_ostfeld/tariffs-tou.csv",
              "data/price_ostfeld/tariffs-it.csv",
              "data/price_ostfeld/tariffs-hu.csv"],
             ["Time-of-Use", "Italy", "Hungary"],
             start=datetime.datetime(2023, 1, 1),
             end=datetime.datetime(2023, 2, 1),
             output_path="outputs/paper_figures/tariffs.png")

# ---------------------------------------------------------------------------
# Case study I
# ---------------------------------------------------------------------------
CASE1_MODELS = (["LP"] + [f"RO_omega=2.5"] + [f"RO_gamma=[8, 10]"] + [f"SP_alpha=[0.95, 0.95]|lambda_cvar=0"]
                + [f"SP_alpha=[0.95, 0.95]|lambda_cvar=1"])
CASE1_LEGEND = [
    ["LP", f"RO_omega=2.5", f"RO_gamma=[8, 10]"],
    [f"SP_alpha=[0.95, 0.95]|lambda_cvar=0", f"SP_alpha=[0.95, 0.95]|lambda_cvar=1"]
]

net = LPNetwork('data/price_ostfeld')
traj = pd.read_csv('outputs/paper/01_jan2023_hu/trajectories.csv')
baseline_traj = pd.read_csv(f'outputs/paper/01_jan2023_hu_oracle/trajectories.csv')
summary = pd.read_csv('outputs/paper/01_jan2023_hu/summary.csv')

g = MPCGraphs(traj, net, inp_file='data/price_ostfeld/price_ostfeld.inp')

# for graphical abstract
g.plot_tank_trajectories(methods=CASE1_MODELS, levels=False, epanet_bounds=True,
                         start_time=datetime.datetime(2023, 1, 21),
                         end_time=datetime.datetime(2023, 2, 1),
                         tank_ids=[2],
                         )

g.plot_por(summary, violations_metric="violated_vol", output_path="outputs/paper_figures/01_por.png",
           filters={
               "omega": [0.5, 1.0, 1.5, 2.0, 2.5],  # remove \omega=3.0 due to many infeasible runs
               "gamma_cost": [8.0],
               "lambda_cvar": [0.0, 1.0]
           })

g.plot_tank_trajectories(methods=CASE1_MODELS, levels=False, epanet_bounds=True,
                         start_time=datetime.datetime(2023, 1, 1),
                         end_time=datetime.datetime(2023, 1, 8),
                         output_path="outputs/paper_figures/01_vol_trajectories.png")

MODELS = (
        [f"SP_alpha=[{a}, {a}]|lambda_cvar=0" for a in [0.8, 0.85, 0.9, 0.95, 0.99]]
        + [f"SP_alpha=[{a}, {a}]|lambda_cvar=1" for a in [0.8, 0.85, 0.9, 0.95, 0.99]]
        + [f"RO_omega={o}" for o in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]
)
L = ([[f"RO_omega={o}" for o in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]],
     [f"SP_alpha=[{a}, {a}]|lambda_cvar=0" for a in [0.8, 0.85, 0.9, 0.95, 0.99]],
     [f"SP_alpha=[{a}, {a}]|lambda_cvar=1" for a in [0.8, 0.85, 0.9, 0.95, 0.99]]
])

g.plot_cumulative_metrics(methods=CASE1_MODELS, output_path="outputs/paper_figures/01_hu_cumulative.png",
                          baseline_trajectories=baseline_traj, legend_rows=CASE1_LEGEND)

g.plot_cumulative_energy_by_price(methods=CASE1_MODELS,
                                  output_path="outputs/paper_figures/01_cum_energy_by_price.png")
g.plot_energy_price_quantile(methods=CASE1_MODELS, n_bins=50, mode='share', legend_rows=CASE1_LEGEND,
                             output_path="")
g.plot_energy_price_quantile(methods=CASE1_MODELS, n_bins=50, mode='cumulative', legend_rows=CASE1_LEGEND,
                             xtick_step=0.1, output_path="outputs/paper_figures/01_cum_energy_price_percentile.png")

g.plot_cost_trajectory(methods=["LP", "RO_omega=2.5", "RO_gamma=[8, 14]", "SP_alpha=[0.95, 0.95]|lambda_cvar=0",
                                "SP_alpha=[0.95, 0.95]|lambda_cvar=1"])
g.plot_price_error_regret(baseline_trajectories=baseline_traj, methods=CASE1_MODELS)

# ---------------------------------------------------------------------------
# Case study II
# ---------------------------------------------------------------------------
CASE2_MODELS = (["LP"] + [f"RO_omega=2.5"]
                + [f"RO_gamma=[8, 14]"]
                + [f"SP_alpha=[0.95, 0.95]|lambda_cvar=0"]
                + [f"SP_alpha=[0.95, 0.95]|lambda_cvar=1"])
CASE2_LEGEND = [
    ["LP", f"RO_omega=2.5", f"RO_gamma=[8, 14]"],
    [f"SP_alpha=[0.95, 0.95]|lambda_cvar=0", f"SP_alpha=[0.95, 0.95]|lambda_cvar=1"]
]

net = LPNetwork('data/two_mains')
traj = pd.read_csv('outputs/paper/02_jan2023_hu_240726/trajectories.csv')
baseline_traj = pd.read_csv(f'outputs/paper/02_jan2023_hu_oracle_280726/trajectories.csv')
summary = pd.read_csv('outputs/paper/02_jan2023_hu_240726/summary.csv')

g = MPCGraphs(traj, net, inp_file='data/two_mains/two_mains.inp')

g.plot_por(summary, violations_metric="violated_vol", output_path="outputs/paper_figures/02_por.png")
g.plot_tank_trajectories(methods=CASE2_MODELS, levels=False, epanet_bounds=True,
                         start_time=datetime.datetime(2023, 1, 1),
                         end_time=datetime.datetime(2023, 1, 8),
                         output_path="outputs/paper_figures/02_vol_trajectories.png")
g.plot_cumulative_metrics(methods=CASE2_MODELS, output_path="outputs/paper_figures/02_hu_cumulative.png",
                          baseline_trajectories=baseline_traj, legend_rows=CASE2_LEGEND)
g.plot_cumulative_energy_by_price(methods=CASE2_MODELS,
                                  output_path="outputs/paper_figures/02_cum_energy_by_price.png")
g.plot_energy_price_quantile(methods=CASE2_MODELS, n_bins=50, mode='cumulative', legend_rows=CASE2_LEGEND,
                             xtick_step=0.1, output_path="outputs/paper_figures/02_cum_energy_price_percentile.png")
g.plot_price_error_regret(baseline_trajectories=baseline_traj, methods=CASE2_MODELS)

# ---------------------------------------------------------------------------
# Tariff / forecast / case-study gradient analysis (tariff_gradient_analysis.py)
# ---------------------------------------------------------------------------
TARIFF_CONFIGS = {
    'TOU': dict(summary='outputs/paper/01_jan2023_tou_180726/summary.csv',
                oracle='outputs/paper/01_jan2023_tou_oracle/summary.csv',
                tariff_path='data/price_ostfeld/tariffs-tou.csv'),
    'IT': dict(summary='outputs/paper/01_jan2023_it_250726/summary.csv',
               oracle='outputs/paper/01_jan2023_it_oracle/summary.csv',
               tariff_path='data/price_ostfeld/tariffs-it.csv'),
    'HU': dict(summary='outputs/paper/01_jan2023_hu/summary.csv',
               oracle='outputs/paper/01_jan2023_hu_oracle/summary.csv',
               tariff_path='data/price_ostfeld/tariffs-hu.csv'),
}

FORECAST_CONFIGS = {
    'Naive': dict(summary='outputs/paper/01_jan2023_hu_naive_220726/summary.csv'),
    'RF': dict(summary='outputs/paper/01_jan2023_hu_rf_230726/summary.csv'),
    'XGB': dict(summary='outputs/paper/01_jan2023_hu/summary.csv'),
}
FORECAST_ORACLE = 'outputs/paper/01_jan2023_hu_oracle/summary.csv'
FORECAST_ORDER = ['Naive', 'RF', 'XGB']

METHODS = [
    'LP',
    'RO_omega=2.5',
    'RO_gamma=[8, 14]',
    'SP_alpha=[0.95, 0.95]|lambda_cvar=0',
    'SP_alpha=[0.95, 0.95]|lambda_cvar=1',
]

CASE_CONFIGS = {
    'Case 1': {
        'TOU': dict(summary='outputs/paper/01_jan2023_tou_180726/summary.csv',
                    oracle='outputs/paper/01_jan2023_tou_oracle/summary.csv',
                    tariff_path='data/price_ostfeld/tariffs-tou.csv'),
        'IT': dict(summary='outputs/paper/01_jan2023_it_250726/summary.csv',
                   oracle='outputs/paper/01_jan2023_it_oracle/summary.csv',
                   tariff_path='data/price_ostfeld/tariffs-it.csv'),
        'HU': dict(summary='outputs/paper/01_jan2023_hu/summary.csv',
                   oracle='outputs/paper/01_jan2023_hu_oracle/summary.csv',
                   tariff_path='data/price_ostfeld/tariffs-hu.csv'),
    },
    'Case 2': {
        'TOU': dict(summary='outputs/paper/02_jan2023_tou_260726/summary.csv',
                    oracle='outputs/paper/02_jan2023_tou_oracle/summary.csv',
                    tariff_path='data/two_mains/tariffs-tou.csv'
                    ),
        'IT': dict(summary='outputs/paper/02_jan2023_it_250726/summary.csv',
                   oracle='outputs/paper/02_jan2023_it_oracle/summary.csv',
                   tariff_path='data/two_mains/tariffs-it.csv'
                   ),
        'HU': dict(summary='outputs/paper/02_jan2023_hu_240726/summary.csv',
                   oracle='outputs/paper/02_jan2023_hu_oracle_280726/summary.csv',
                   tariff_path='data/two_mains/tariffs-hu.csv'
                   ),
    },
}

table = build_gradient_table(tariff_configs=TARIFF_CONFIGS, methods=METHODS)
table.to_csv('outputs/paper_figures/tariff_summary.csv', index=False)

graphical_abstract(table, category_col='tariff',
                   category_order=['TOU', 'IT', 'HU'], xticklabels=['TOU', 'IT', 'HU'],
                   methods=METHODS, vol_ylim=(-25, None), xlabel='Tariffs Dataset')

plot_tariff_gradient_bars(table, output_path='outputs/paper_figures/tariff_gradient_bars.png')

forecast_table = build_forecast_table(forecast_configs=FORECAST_CONFIGS, oracle_summary=FORECAST_ORACLE,
                                      methods=METHODS)
forecast_table.to_csv('outputs/paper_figures/forecast_gradient.csv', index=False)

plot_forecast_gradient_bars(forecast_table, forecast_order=FORECAST_ORDER,
                            output_path='outputs/paper_figures/forecast_gradient_bars.png')
case_table = build_case_study_table(case_configs=CASE_CONFIGS, methods=METHODS)
case_table.to_csv('outputs/paper_figures/case_study_summary.csv', index=False)

export_case_study_latex(
    case_table, tariffs=['HU'], methods=METHODS,
    caption='Case study comparison under the HU tariff.', label='tab:case_study_hu',
    output_path='outputs/paper_figures/case_study_summary_hu.tex')
export_case_study_latex(
    case_table, tariffs=['HU', 'IT', 'TOU'], methods=METHODS,
    caption='Case study comparison under the IT and TOU tariffs.', label='tab:case_study_supp',
    output_path='outputs/paper_figures/case_study_summary_supp.tex')

# ---------------------------------------------------------------------------
# Computational analysis (computational_analysis.py)
# ---------------------------------------------------------------------------
COMPUTATIONAL_OUTPUT_DIRS = {
    "1": "outputs/paper/01_jan2023_hu",
    "2": "outputs/paper/02_jan2023_hu_240726"
}

computational_table = build_computational_table(COMPUTATIONAL_OUTPUT_DIRS, group_by=["case_study", "group"],
                                                 max_infeasible=50)
computational_table.to_csv('outputs/paper_figures/computational_analysis.csv', index=False)
to_latex_table(computational_table, output_path='outputs/paper_figures/computational_analysis.tex', resizebox=False)
to_latex_table(computational_table, output_path='outputs/paper_figures/computational_analysis_fit.tex',
               resizebox=True)

plot_binarization_sensitivity(
    'outputs/paper/02_jan2023_hu_sa_nbinary_280726',
    output_path='outputs/paper_figures/computational_binarization_sensitivity.png',
    max_infeasible=50
)

BINARIZATION_LEGEND_ROWS = [
    ["LP"],
    [f"RO_omega=2.5", f"RO_gamma=[8, 14]"],
    [f"SP_alpha=[0.9, 0.9]|lambda_cvar=0", f"SP_alpha=[0.95, 0.95]|lambda_cvar=0"],
    [f"SP_alpha=[0.9, 0.9]|lambda_cvar=1", f"SP_alpha=[0.95, 0.95]|lambda_cvar=1"]
]
plot_binarization_cost_bars(
    'outputs/paper/02_jan2023_hu_sa_nbinary_280726',
    output_path='outputs/paper_figures/computational_binarization_cost_bars.png',
    max_infeasible=50,
    legend_rows=BINARIZATION_LEGEND_ROWS
)

plt.show()
