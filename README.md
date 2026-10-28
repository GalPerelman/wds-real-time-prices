# Water Distribution System Operation as a Physical Hedge in Volatile Electricity Markets

Uncertainty-aware optimization for water distribution system pump scheduling under volatile electricity prices and water demand.

**Authors:** Kristina Korder, Gal Perelman, Elad Salomons, Avi Ostfeld, Pu Li

## Abstract

Water distribution systems (WDSs) are increasingly exposed to volatile electricity prices as power markets become more dynamic under growing renewable generation. At the same time, water storage capacity can be employed to decouple pumping from water consumption, providing operational flexibility to hedge against electricity price volatility. On the other hand, sufficient storage margins must be maintained to ensure reliable water supply under uncertain demand.
This study evaluates WDS operation from this dual risk-management perspective using a rolling-horizon optimization framework under both water demand and electricity price uncertainties. Different problem formulations are studied and compared, including deterministic predict-then-optimize (PTO), chance-constrained optimization (CCO), CCO with conditional value-at-risk (CVaR), and robust optimization (RO), evaluated across two WDS case studies and increasingly volatile electricity price regimes. The results show that uncertainty-aware operation can significantly reduce storage-bound violations at modest realized cost premiums, generally below 1% for the better-performing formulations, with some cases yielding negative premiums. Hedging performance depends strongly on how uncertainty is represented, with correlation-aware formulations achieving a more effective balance between reliability and exploitation of favorable electricity prices. Forecast sensitivity analysis further shows that lower prediction error does not necessarily improve closed-loop optimization performance. The findings demonstrate that WDS storage should be considered not only as hydraulic infrastructure but also as an energy-flexibility asset for managing WDSs under increasingly volatile electricity prices and water demand uncertainty.

## Overview

This repository implements pump scheduling optimization for water distribution systems (WDS)
operating under uncertain electricity prices and water demand. It compares:

- **Deterministic** scheduling (LP/MILP)
- **Robust optimization (RO)** with ellipsoidal and budget (BOU) uncertainty sets
- **Stochastic programming (SP)** with chance constraints and Expected-CVaR objective
- **Rolling-horizon Model Predictive Control (MPC)**, driven by ML-based (XGBoost,
  random forest) or Naive forecasts

Pump schedules are evaluated both analytically (LP relaxation) and via step-by-step **EPANET**
hydraulic simulation for realistic feasibility and energy accounting.

## Case Studies

| | Network | Tanks | Pump stations | Configs |
|---|---|---|---|---|
| Case Study I | `price_ostfeld` | 3 | 4 | `configs/01/` |
| Case Study II | `two_mains` | 6 | 7 | `configs/02/` |


## Installation

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Unix

pip install -r requirements.txt
```

**Solver:** Gurobi (primary, requires a license) with Clarabel / ECOS / Mosek / SCS / OSQP as
CVXPY fallbacks.

## Reproducing an Experiment

Each result in the paper is defined by a YAML config under `configs/01/` (Case Study I) or
`configs/02/` (Case Study II). Run one with:

```bash
python main.py --config configs/02/jan2023_hu.yaml
```

Multiple configs can be run in one invocation:

```bash
python main.py --config configs/02/jan2023_hu.yaml configs/02/jan2023_it.yaml
```

See `configs/examples/` for minimal, documented configs demonstrating each optimization method
and sweep structure.

