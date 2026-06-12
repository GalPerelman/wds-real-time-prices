"""
Simple YAML Configuration for WDS Optimization Experiments
"""

from itertools import product
from pathlib import Path
import yaml


class Config(dict):
    """Dictionary with attribute access for cleaner syntax."""

    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                return Config(value)
            return value
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def load_config(config_path):
    """Load YAML config file into a Config object."""
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return Config(raw)


def get_model_class(opt_method):
    """Map string name to optimization class (case-insensitive)."""
    from opt import BaseOptModel, RO, SP
    mapping = {'baseoptmodel': BaseOptModel, 'lp': BaseOptModel, 'ro': RO, 'sp': SP, 'saa': SP}
    return mapping[opt_method.lower()]


def expand_methods(methods):
    """
    Expand method configs with sweeps into a flat list.

    Used by both improved_por() (POR) and run_mpc_sweep() (MPC sweep).
    The optimization model is identified by 'opt_method' in all config files.

    Input format (from YAML):
        - group: "RO-Ellipsoid"
          opt_method: "ro"
          params: {uncertainty_set: "ellipsoid"}
          sweep: {omega: [0.5, 1.0, 1.5]}

    Output format:
        [{'label': 'omega=0.5', 'group': 'RO-Ellipsoid', 'opt_method': 'ro', 'params': {...}}, ...]
    """
    expanded = []
    for m in methods:
        group = m.get('group', m.get('label', 'method'))
        opt_method = m.get('opt_method', 'lp')
        base_params = m.get('params', {})
        sweep = m.get('sweep')

        if sweep:
            param_names = list(sweep.keys())
            param_values = list(sweep.values())
            for combo in product(*param_values):
                params = {**base_params, **dict(zip(param_names, combo))}
                label_parts = '|'.join(f'{k}={v}' for k, v in zip(param_names, combo))
                expanded.append({
                    'label':      f'{opt_method.upper()}_{label_parts}',
                    'group':      group,
                    'opt_method': opt_method,
                    'params':     params,
                })
        else:
            expanded.append({
                'label':      m.get('label', group),
                'group':      group,
                'opt_method': opt_method,
                'params':     base_params
            })
    return expanded
