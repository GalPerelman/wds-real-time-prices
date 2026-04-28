import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd
from matplotlib import ticker as mtick
from matplotlib.gridspec import GridSpec

# Global palette for optimization methods.
# 'color'  - representative fixed color for single-point/static plots
# 'cmap'   - colormap for multi-parameter sweep shading
# 'marker' - shared marker style
# Per-group styles for all MPC plots. Keys match the 'group' field in experiment configs.
# 'color'  - single representative color used in plot_por
# 'cmap'   - colormap used in plot_tank_trajectories to shade individual sweep variants
# 'marker' - marker style used in plot_por
# All three fields are required for every entry (no fallback).
GROUP_STYLES = {
    'LP':            {'color': 'k',        'cmap': 'Greys',    'marker': 'X'},
    'RO-Ellipsoid':  {'color': 'C0',  'cmap': 'Blues',    'marker': 'o'},
    'RO-BOU':     {'color': '#368399',  'cmap': 'Greens',  'marker': 'D'},
    'RO-BOU-1':     {'color': '#368399',  'cmap': 'Greens',  'marker': 'D'},
    'RO-BOU-2':     {'color': '#8670FF',  'cmap': 'Purples',  'marker': 'D'},
    'CC-SP':         {'color': '#d13b2c',  'cmap': 'Reds',     'marker': 's'},
    'CC-SP-relaxed': {'color': 'C1',  'cmap': 'Oranges',  'marker': '^'},
}


def to_latex_label(s):
    prefix, rest = s.split('_', 1)  # "RO", "omega=1"
    var, val = rest.split('=')  # "omega", "1"

    # capitalize Greek letter if needed
    if var.lower() == 'omega':
        var = r'\Omega'

    return rf"${prefix}_{{{var}={val}}}$"


def build_label_colors(label_to_group):
    """
    Assign one color per label using GROUP_STYLES.

    label_to_group: ordered dict {label_str: group_str}
        e.g. {'RO_omega=0.5': 'RO-Ellipsoid', 'RO_omega=1.0': 'RO-Ellipsoid',
              'SP_alpha=0.9': 'CC-SP'}

    Within each group, multiple labels get shaded across the group's cmap [0.45, 0.85].
    A single label in a group gets the group's fixed 'color'.

    Returns {label: rgba_color}.
    """
    groups_ordered = list(dict.fromkeys(label_to_group.values()))
    colors = {}
    for group in groups_ordered:
        labels_in_group = [l for l, g in label_to_group.items() if g == group]
        style = GROUP_STYLES[group]
        if len(labels_in_group) == 1:
            colors[labels_in_group[0]] = mcolors.to_rgba(style['color'])
        else:
            cmap = mpl.colormaps[style['cmap']]
            shades = np.linspace(0.45, 0.85, len(labels_in_group))
            for i, label in enumerate(labels_in_group):
                colors[label] = cmap(shades[i])
    return colors


class OptGraphs:
    def __init__(self, opt_model):
        self.opt_model = opt_model
        self.x_fsp_value = opt_model.x_fsp_value
        self.x_vsp_value = opt_model.x_vsp_value
        self.t = self.opt_model.t

    def tank_volume(self, tank_id, plot_dem=False, plot_tariff=False, sample=None, ax=None):
        if sample is not None and sample.shape[1] > 1:
            alpha = 0.1
        else:
            alpha = 1.0

        vol = self.opt_model.network.get_tank_vol(t=self.opt_model.t, tank_id=tank_id,
                                                  x_fsp=self.x_fsp_value, x_vsp=self.x_vsp_value, sample=sample,
                                                  include_init=True)
        if ax is None:
            fig, ax = plt.subplots()

        tank_data = self.opt_model.network.tanks.loc[tank_id]

        ax.plot(vol, c='C0', alpha=alpha)
        demands = self.opt_model.network.get_tank_demand(self.opt_model.t, tank_id)

        if plot_dem:
            ax.bar(range(1, len(demands) + 1), demands, edgecolor='k', zorder=5)

        if plot_tariff:
            tariff = self.opt_model.network.tariffs['tariff'].values[:self.opt_model.t]
            ax.bar(range(1, len(tariff) + 1), tariff, edgecolor='k', zorder=5)

        ax.hlines(tank_data['init_vol'], 0, self.t + 1, colors='k', linestyles='--')
        ax.hlines(tank_data['min_vol'], 0, self.t + 1, colors='red')
        ax.hlines(tank_data['max_vol'], 0, self.t + 1, colors='red')
        ax.grid(zorder=0)
        return ax

    def all_tanks(self, plot_dem=False, plot_tariff=False, sample=None, ncols=None):
        if plot_tariff:
            j = 1
        else:
            j = 0

        if ncols is None:
            ncols = max(1, int(math.ceil(math.sqrt(self.opt_model.network.n_tanks))))
            nrows = max(1, int(math.ceil(self.opt_model.network.n_tanks / ncols))) + j
        else:
            nrows = self.opt_model.network.n_tanks + j

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 7))
        axes = axes.ravel()
        for i, (tank_id, tank_data) in enumerate(self.opt_model.network.tanks.iterrows()):
            if tank_id == 9:
                continue

            if sample is not None:
                dem_sample = sample[self.t + i * self.t: self.t + i * self.t + self.t:, :]
            else:
                dem_sample = None

            self.tank_volume(tank_id, ax=axes[i], plot_dem=plot_dem, plot_tariff=False, sample=dem_sample)
            axes[i].set_title(f"Tank {tank_id}")
            axes[i].set_ylabel('Volume ($m^3$)')

        if plot_tariff:
            cost_sample = sample[: self.t:, :]
            axes[-1].plot(cost_sample, alpha=0.1, c='C0')
            axes[-1].plot(self.opt_model.network.tariffs['tariff'].values[:self.t], c='k')
            axes[-1].grid(zorder=0)
            # axes[-1].bar(range(self.t), self.opt_model.network.tariffs['tariff'].values[:self.t], edgecolor='k', alpha=0.6)
            axes[-1].set_ylabel('Tariff (€/MWh)')

        axes[-1].set_xlabel('Time (Hr)')
        fig.align_ylabels()
        fig.subplots_adjust(left=0.12, right=0.95, hspace=0.4, top=0.95, bottom=0.08)


def plot_mat(mat, norm=False, t=24):
    fig = plt.figure()
    if norm:
        mat = (mat - mat.min()) / (mat.max() - mat.min())

    mat_norm = max(abs(mat.min()), abs(mat.max()))
    im = plt.imshow(mat, cmap="RdYlBu_r")
    ax = plt.gca()

    ax.tick_params(which='minor', bottom=False, left=False)
    cbar = plt.colorbar(im, ticks=mtick.AutoLocator())

    # Major ticks
    ax.set_xticks(np.arange(-0.5, mat.shape[0], t))
    ax.set_yticks(np.arange(-0.5, mat.shape[0], t))
    ax.set_xticklabels(np.arange(0, mat.shape[0] + t, t))
    ax.set_yticklabels(np.arange(0, mat.shape[0] + t, t))

    # Gridlines based on minor ticks
    ax.grid(which='major', color='k', linestyle='-', linewidth=0.8)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.4, alpha=0.4)

    ax.set_xticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    fig.subplots_adjust(top=0.9, bottom=0.13, left=0.055, right=0.9, hspace=0.2, wspace=0.2)
    return ax


def series_hourly_distribution(df: pd.DataFrame, col: str):
    df = df.copy()
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    df['day_of_year'] = df.index.dayofyear
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    columns = df.columns.difference(['day', 'hour', 'day_of_year', 'weekday', 'month'])

    fig, axes = plt.subplots(nrows=3, figsize=(8, 6))

    pivot_temp = df.pivot_table(values=col, index=df.index.date, columns=df.index.hour)
    pivot_temp.boxplot(ax=axes[0], positions=range(24))
    axes[0].set_ylabel("Tariff (€/MWh)")
    axes[0].grid(False)
    axes[0].set_xlabel('Hour of the day')

    axes[1].plot(pivot_temp.values.T, c='C0', alpha=0.1)
    axes[1].set_ylabel("Tariff (€/MWh)")
    axes[1].grid()
    axes[1].set_xlabel('Hour of the day')

    axes[2].hist(df[col], bins=25, edgecolor='k', alpha=0.5)
    axes[2].set_ylabel(col)
    axes[2].set_xlabel("Count")

    fig.align_ylabels()
    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.9, hspace=0.15)


def plot_predicted_vs_realized(panels, figsize=None):
    """
    Plot predicted vs realized time series for one or more signals.

    panels: list of dicts, each with:
        'time'      - array-like of timestamps
        'predicted' - array-like of predicted values
        'realized'  - array-like of realized values
        'ylabel'    - str, y-axis label

    Returns fig, axes.
    """
    n = len(panels)
    fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=figsize or (6, 5))
    if n == 1:
        axes = [axes]

    for i, panel in enumerate(panels):
        ax = axes[i]
        ax.plot(panel['time'], panel['realized'], label='Realized', color='C0')
        ax.plot(panel['time'], panel['predicted'], label='Predicted', color='C1')
        ax.set_ylabel(panel['ylabel'])
        if i == 0:
            ax.legend(loc='upper right')
        mae = np.mean(np.abs(np.asarray(panel['predicted']) - np.asarray(panel['realized'])))
        ax.text(0.02, 0.82, f'MAE={mae:.2f}', transform=ax.transAxes, horizontalalignment='left')

    axes[-1].set_xlabel('Time')
    fig.align_ylabels()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.08)
    return fig, axes


def plot_mpc_forecast_accuracy(trajectories, method=None, figsize=None):
    """
    Plot predicted vs realized demands and price from MPC sweep trajectories.

    All methods are assumed to share the same forecaster, so any one method's
    data is representative. method defaults to the first label in trajectories.
    """
    if method is None:
        method = trajectories['label'].iloc[0]
    df = trajectories[trajectories['label'] == method]
    time = pd.to_datetime(df['time'])

    tank_ids = [c.replace('predicted_demand_', '')
                for c in df.columns if c.startswith('predicted_demand_')]

    panels = []
    for tank_id in tank_ids:
        panels.append({
            'time': time,
            'predicted': df[f'predicted_demand_{tank_id}'].values,
            'realized': df[f'realized_demand_{tank_id}'].values,
            'ylabel': f'Demand {tank_id}\n($m^3/h$)',
        })
    panels.append({
        'time': time,
        'predicted': df['predicted_tariff'].values,
        'realized': df['realized_tariff'].values,
        'ylabel': 'Tariff\n(€/MWh)',
    })

    return plot_predicted_vs_realized(panels, figsize=figsize)


class MPCGraphs:
    def __init__(self, trajectories, net):
        self.trajectories = trajectories
        self.net = net

    def plot_hydraulic_accuracy(self, methods=None):
        all_labels = self.trajectories['label'].unique().tolist()
        if methods is None:
            methods = all_labels
        n_methods = len(methods)
        n_tanks = self.net.n_tanks
        tank_ids = list(self.net.tanks.index)

        fig, axes = plt.subplots(nrows=n_methods, ncols=n_tanks, figsize=(3.2 * n_tanks, 2 * n_methods), sharex=True)
        if n_methods == 1:
            axes = axes[np.newaxis, :]
        if n_tanks == 1:
            axes = axes[:, np.newaxis]

        available = self.trajectories['label'].unique().tolist()
        for row, label in enumerate(methods):
            df = self.trajectories[self.trajectories['label'] == label]
            if df.empty:
                raise ValueError(f"Label '{label}' not found. Available: {available}")
            for col, tank_id in enumerate(tank_ids):
                ax = axes[row, col]
                opt_col = f'opt_vol_tank_{tank_id}'
                epanet_col = f'epanet_vol_tank_{tank_id}'
                ax.plot(df[opt_col].values, marker='o', color='C0', label='Opt', markeredgecolor='k',
                        markersize=0, alpha=1)
                if epanet_col in df.columns:
                    ax.plot(df[epanet_col].values, marker='o', color='C1', label='EPANET', markeredgecolor='k',
                            markersize=0, alpha=1)
                tank_data = self.net.tanks.loc[tank_id]
                ax.axhline(tank_data['min_vol'], color='k', linestyle='--')
                ax.axhline(tank_data['max_vol'], color='k', linestyle='--')
                ax.grid()
                if row == 0:
                    ax.set_title(f'Tank {tank_id}')
                if col == 0:
                    model = df['model'].iloc[0].upper()
                    prefix = '' if model == label.upper() else f'{model} '
                    ax.set_ylabel(f'{prefix}{label}\nVolume ($m^3$)')
                if row == n_methods - 1:
                    ax.set_xlabel('Time (Hr)')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.1, 1), loc='upper left', bbox_transform=fig.transFigure, ncol=2)
        fig.align_ylabels()
        fig.subplots_adjust(left=0.11, right=0.97, hspace=0.2, wspace=0.25, top=0.88, bottom=0.08)
        return fig, axes

    def plot_energy_accuracy(self, methods=None):
        all_labels = self.trajectories['label'].unique().tolist()
        if methods is None:
            methods = all_labels

        pump_cols = self.net.fsp.columns[self.net.fsp.columns.str.startswith('pump_')]
        pump_membership = self.net.fsp[pump_cols].copy()
        pump_membership.columns = pump_membership.columns.str.removeprefix('pump_')
        pump_ids = list(pump_membership.columns)
        n_pumps = len(pump_ids)
        n_methods = len(methods)

        fig, axes = plt.subplots(nrows=n_methods, ncols=n_pumps,
                                 figsize=(3.2 * n_pumps, 2 * n_methods), sharex=True)
        if n_methods == 1:
            axes = axes[np.newaxis, :]
        if n_pumps == 1:
            axes = axes[:, np.newaxis]

        for row, label in enumerate(methods):
            df = self.trajectories[self.trajectories['label'] == label]
            if df.empty:
                raise ValueError(f"Label '{label}' not found. Available: {all_labels}")

            # Disaggregate combo energy to physical pump level
            opt_energy_pump = {}
            for pump_id in pump_ids:
                combos_with_pump = pump_membership.index[pump_membership[pump_id] > 0]
                opt_energy_pump[pump_id] = sum(
                    df[f'opt_energy_{c}'].values for c in combos_with_pump
                )

            model = df['model'].iloc[0].upper()
            prefix = '' if model == label.upper() else f'{model} '

            for col, pump_id in enumerate(pump_ids):
                ax = axes[row, col]
                ax.plot(opt_energy_pump[pump_id], marker='o', color='C0', label='Opt',
                        markeredgecolor='k', markersize=0)
                epanet_col = f'epanet_energy_{pump_id}'
                if epanet_col in df.columns:
                    ax.plot(df[epanet_col].values, marker='o', color='C1', label='EPANET',
                            markeredgecolor='k', markersize=0)
                ax.grid()
                if row == 0:
                    ax.set_title(f'Pump {pump_id}')
                if col == 0:
                    ax.set_ylabel(f'{prefix}{label}\nEnergy (kWh)')
                if row == n_methods - 1:
                    ax.set_xlabel('Time (Hr)')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.1, 1), loc='upper left', bbox_transform=fig.transFigure, ncol=2)
        fig.align_ylabels()
        fig.subplots_adjust(left=0.11, right=0.97, hspace=0.2, wspace=0.25, top=0.88, bottom=0.08)
        return fig, axes

    def plot_hydraulic_scatter(self, methods=None):
        all_labels = self.trajectories['label'].unique().tolist()
        if methods is None:
            methods = all_labels

        df = self.trajectories[self.trajectories['label'].isin(methods)]
        tank_ids = list(self.net.tanks.index)

        opt_vals, epanet_vals = [], []
        for tank_id in tank_ids:
            opt_col = f'opt_vol_tank_{tank_id}'
            epanet_col = f'epanet_vol_tank_{tank_id}'
            if epanet_col in df.columns:
                opt_vals.append(df[opt_col].values)
                epanet_vals.append(df[epanet_col].values)

        opt_vals = np.concatenate(opt_vals)
        epanet_vals = np.concatenate(epanet_vals)
        mae = np.mean(np.abs(opt_vals - epanet_vals))

        fig, ax = plt.subplots()
        ax.scatter(opt_vals, epanet_vals, color='C0', alpha=0.5, edgecolors='k', linewidths=0.4, zorder=5)

        lims = [min(opt_vals.min(), epanet_vals.min()) * 0.4, max(opt_vals.max(), epanet_vals.max()) * 1.1]
        ax.plot(lims, lims, color='k', linestyle='--', linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel('Opt Volume ($m^3$)')
        ax.set_ylabel('EPANET Volume ($m^3$)')
        ax.text(0.04, 0.95, f'MAE = {mae:.1f} $m^3$', transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='grey', linewidth=1, alpha=0.5))
        ax.set_axisbelow(True)
        ax.grid()
        ax.set_aspect('equal')
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15)
        return fig, ax

    def plot_tank_trajectories(self, methods=None, show_opt=False):
        """
        Plot tank volume trajectories over the full MPC experiment.

        One row per tank plus a shared tariff panel at the bottom.
        All methods are overlaid on each tank subplot, colored by group
        (each group gets its color from GROUP_STYLES; all sweep variants
        within a group share the same color).

        Args:
            methods:  list of labels to include; None = all labels
            show_opt: if True and EPANET columns are present, overlay the
                      optimizer (LP mass-balance) volumes as dashed lines.
        """
        all_labels = self.trajectories['label'].unique().tolist()
        if methods is None:
            methods = all_labels

        tank_ids = list(self.net.tanks.index)
        n_tanks = len(tank_ids)

        label_to_group = {}
        for label in methods:
            row = self.trajectories[self.trajectories['label'] == label]
            if not row.empty:
                label_to_group[label] = row['group'].iloc[0]
        label_colors = build_label_colors(label_to_group)

        has_epanet = f'epanet_vol_tank_{tank_ids[0]}' in self.trajectories.columns

        n_rows = n_tanks + 1
        height_ratios = [2] * n_tanks + [1]
        fig = plt.figure(figsize=(6, 1 * n_tanks + 1))
        gs = GridSpec(n_rows, 1, height_ratios=height_ratios, hspace=0.15, figure=fig)
        tank_axes = [fig.add_subplot(gs[0])]
        for i in range(1, n_tanks):
            tank_axes.append(fig.add_subplot(gs[i], sharex=tank_axes[0]))
        tariff_ax = fig.add_subplot(gs[-1], sharex=tank_axes[0])

        for ax_idx, tank_id in enumerate(tank_ids):
            ax = tank_axes[ax_idx]
            tank_data = self.net.tanks.loc[tank_id]
            vol_col_key = f'epanet_vol_tank_{tank_id}' if has_epanet else f'opt_vol_tank_{tank_id}'
            opt_col_key = f'opt_vol_tank_{tank_id}'

            for label in methods:
                df = self.trajectories[self.trajectories['label'] == label]
                if df.empty:
                    raise ValueError(f"Label '{label}' not found. Available: {all_labels}")
                color = label_colors[label]
                time = pd.to_datetime(df['time'])
                vol = df[vol_col_key].values

                zorder = 5 if label in ['lp', 'LP'] else 3
                ax.plot(time, vol, color=color, linewidth=1.2, label=label, zorder=zorder)

                if show_opt and has_epanet:
                    ax.plot(time, df[opt_col_key].values, color=color, linewidth=0.8,
                            linestyle='--', alpha=0.55, zorder=2)

                # Violation markers: 'x' where volume crosses real bounds
                viol_mask = (vol < tank_data['min_vol']) | (vol > tank_data['max_vol'])
                if viol_mask.any():
                    ax.scatter(time[viol_mask], vol[viol_mask], color=color,
                               marker='o', s=50, linewidths=1.5, zorder=5, facecolor="none")

            ax.axhline(tank_data['min_vol'], color='k', linestyle='--', linewidth=0.9, zorder=1)
            ax.axhline(tank_data['max_vol'], color='k', linestyle='--', linewidth=0.9, zorder=1)
            ax.set_ylabel(f'Tank {tank_id} ($m^3$)')
            ax.grid(zorder=0)
            ax.set_axisbelow(True)
            plt.setp(ax.get_xticklabels(), visible=False)

        # Tariff panel — realized and predicted are identical across methods; use first
        ref_df = self.trajectories[self.trajectories['label'] == methods[0]]
        ref_time = pd.to_datetime(ref_df['time'])
        tariff_ax.plot(ref_time, ref_df['realized_tariff'].values,
                       color='k', linewidth=1.2, label='Realized')
        tariff_ax.plot(ref_time, ref_df['predicted_tariff'].values,
                       color='k', linewidth=0.9, linestyle='--', alpha=0.5, label='Predicted')
        tariff_ax.set_ylabel('Tariff\n(€/MWh)')
        tariff_ax.grid(zorder=0)
        tariff_ax.set_axisbelow(True)
        tariff_ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
        tariff_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        tariff_ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        fig.autofmt_xdate(rotation=0, ha='center')

        # Main legend: methods + opt dashed line + bounds — no tariff (handled locally above)
        from matplotlib.lines import Line2D
        handles, labels_leg = tank_axes[0].get_legend_handles_labels()
        if show_opt and has_epanet:
            handles += [Line2D([0], [0], color='grey', linewidth=1.8, linestyle='--')]
            labels_leg += ['Opt (LP)']
        handles += [Line2D([0], [0], color='k', linewidth=1.8, linestyle='--')]
        labels_leg += ['Min/Max bounds']

        ncol = min(len(handles), 5)
        nrow = math.ceil(len(handles) / ncol)
        idx = list(range(len(handles))) + [-1] * (nrow * ncol - len(handles))
        order = [i for i in np.array(idx).reshape(nrow, ncol).T.flatten() if i >= 0]
        handles = [handles[i] for i in order]
        labels_leg = [labels_leg[i] for i in order]

        leg = fig.legend(handles, labels_leg, loc='upper left', bbox_to_anchor=(0.095, 0.96),
                         bbox_transform=fig.transFigure, ncol=ncol, fontsize=9,
                         framealpha=0.9)
        for line in leg.get_lines():
            line.set_linewidth(2.2)

        fig.subplots_adjust(left=0.1, right=0.97, top=0.87, bottom=0.08)
        return fig, tank_axes, tariff_ax

    def plot_cumulative_metrics(self, methods=None):
        """
        Three stacked subplots with shared x-axis:
          - Cumulative actual cost (EUR), one line per method label
          - Cumulative total violated volume (m3) — sum of min and max violations
          - Realized and predicted tariff reference panel (smaller)
        """
        all_labels = self.trajectories['label'].unique().tolist()
        if methods is None:
            methods = all_labels

        label_to_group = {}
        for label in methods:
            row = self.trajectories[self.trajectories['label'] == label]
            if not row.empty:
                label_to_group[label] = row['group'].iloc[0]
        label_colors = build_label_colors(label_to_group)

        has_epanet = 'epanet_cost' in self.trajectories.columns
        prefix = 'epanet' if has_epanet else 'opt'

        # Detect violation schema: new (split min/max) vs old (combined)
        new_schema = f'{prefix}_min_violated_vol' in self.trajectories.columns

        def get_violated_vol(df):
            if new_schema:
                return df[f'{prefix}_min_violated_vol'].values + df[f'{prefix}_max_violated_vol'].values
            return df[f'{prefix}_violated_vol'].values

        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(3, 1, height_ratios=[1, 1, 0.5], hspace=0.15, figure=fig)
        ax_cost = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax_cost)
        ax_tariff = fig.add_subplot(gs[2], sharex=ax_cost)

        for label in methods:
            df = self.trajectories[self.trajectories['label'] == label]
            if df.empty:
                raise ValueError(f"Label '{label}' not found. Available: {all_labels}")
            color = label_colors[label]
            time = pd.to_datetime(df['time'])

            if has_epanet:
                cost_per_step = df['epanet_cost'].values
            else:
                opt_energy_cols = [c for c in df.columns if c.startswith('opt_energy_')]
                total_opt_energy = df[opt_energy_cols].sum(axis=1)
                cost_per_step = (total_opt_energy * df['realized_tariff'] / 1000).values

            try:
                label = to_latex_label(label)
            except ValueError:
                pass
            ax_cost.plot(time, np.cumsum(cost_per_step), color=color, linewidth=1.2, label=label)
            ax_vol.plot(time, np.cumsum(get_violated_vol(df)), color=color, linewidth=1.2)

        ref_df = self.trajectories[self.trajectories['label'] == methods[0]]
        ref_time = pd.to_datetime(ref_df['time'])
        ax_tariff.plot(ref_time, ref_df['realized_tariff'].values,
                       color='k', linewidth=1.2, label='Realized')
        ax_tariff.plot(ref_time, ref_df['predicted_tariff'].values,
                       color='k', linewidth=0.9, linestyle='--', alpha=0.5, label='Predicted')
        ax_tariff.set_ylabel('Tariff\n(€/MWh)')
        ax_tariff.grid(zorder=0)
        ax_tariff.set_axisbelow(True)
        ax_tariff.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
        ax_tariff.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_tariff.legend(loc='upper right', fontsize=8, framealpha=0.9)
        fig.autofmt_xdate(rotation=0, ha='center')

        ax_cost.set_ylabel('Cumulative\nCost (€)')
        ax_cost.grid(zorder=0)
        ax_cost.set_axisbelow(True)
        plt.setp(ax_cost.get_xticklabels(), visible=False)

        ax_vol.set_ylabel('Cumulative\nViolated Vol. ($m^3$)')
        ax_vol.grid(zorder=0)
        ax_vol.set_axisbelow(True)
        plt.setp(ax_vol.get_xticklabels(), visible=False)

        handles, labels_leg = ax_cost.get_legend_handles_labels()
        ncol = min(len(handles), 5)
        nrow = math.ceil(len(handles) / ncol)
        idx = list(range(len(handles))) + [-1] * (nrow * ncol - len(handles))
        order = [i for i in np.array(idx).reshape(nrow, ncol).T.flatten() if i >= 0]
        handles = [handles[i] for i in order]
        labels_leg = [labels_leg[i] for i in order]

        leg = fig.legend(handles, labels_leg, loc='upper left', bbox_to_anchor=(0.095, 0.96),
                         bbox_transform=fig.transFigure, ncol=ncol, fontsize=9, framealpha=0.9)
        for line in leg.get_lines():
            line.set_linewidth(2.2)

        fig.subplots_adjust(left=0.12, right=0.97, top=0.87, bottom=0.08)
        return fig, (ax_cost, ax_vol, ax_tariff)

    def plot_por(self, summary, violations_metric="violated_vol"):
        """
        Plot cost vs. violations (POR curve) for all method groups.

        summary:           DataFrame with 'group', 'actual_cost', and the
                           violations_metric column.
        violations_metric: column name from summary — 'min_violated_vol', 'max_violated_vol',
                           'violated_vol', or 'constraint_satisfaction_rate'

        Colors and markers are taken from GROUP_STYLES keyed by group name.
        All groups present in the summary must be defined in GROUP_STYLES.
        """
        summary = summary.copy()
        if 'violated_vol' not in summary.columns and \
                'min_violated_vol' in summary.columns and 'max_violated_vol' in summary.columns:
            summary['violated_vol'] = summary['min_violated_vol'] + summary['max_violated_vol']

        fig, ax = plt.subplots(figsize=(6, 5))

        for group, grp_df in summary.groupby('group', sort=False):
            style = GROUP_STYLES[group]
            ax.scatter(grp_df['actual_cost'], grp_df[violations_metric],
                    color=style['color'], marker=style['marker'], s=40,
                    label=group, edgecolor='k')

        ylabels = {
            "violated_vol":                 "Violated Volume ($m^3$)",
            "min_violated_vol":             "Service Disruption ($m^3$)",
            "max_violated_vol":             "Overflow Volume ($m^3$)",
            "constraint_satisfaction_rate": "Constraint Satisfaction Rate",
        }
        ax.set_xlabel('Actual Cost (€)')
        ax.set_ylabel(ylabels.get(violations_metric, violations_metric))
        ax.legend()
        ax.grid()
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.1)
        return fig, ax

    def plot_price_responsiveness(self, methods=None, bin_width=25):
        """
        Grouped boxplot of total EPANET energy vs realized tariff bins.

        The x-axis is divided into fixed-width price bins. Within each bin,
        one box per method label is drawn side by side, colored by group.
        Shows whether the scheduler shifts consumption to cheap hours.
        Requires epanet_energy_* columns in trajectories.

        Args:
            methods:   list of labels to include; None = all labels
            bin_width: width of each price bin in EUR/MWh (default 25)
        """
        energy_cols = [c for c in self.trajectories.columns if c.startswith('epanet_energy_')]
        if not energy_cols:
            raise ValueError("No epanet_energy_* columns found in trajectories.")

        all_labels = self.trajectories['label'].unique().tolist()
        if methods is None:
            methods = all_labels

        label_to_group = {}
        for label in methods:
            row = self.trajectories[self.trajectories['label'] == label]
            if not row.empty:
                label_to_group[label] = row['group'].iloc[0]
        label_colors = build_label_colors(label_to_group)

        traj_sel = self.trajectories[self.trajectories['label'].isin(methods)]
        price_min = np.floor(traj_sel['realized_tariff'].min() / bin_width) * bin_width
        price_max = np.ceil(traj_sel['realized_tariff'].max() / bin_width) * bin_width
        edges = np.arange(price_min, price_max + bin_width, bin_width)
        bin_labels = [f"{int(e)}-{int(e + bin_width)}" for e in edges[:-1]]
        n_bins = len(bin_labels)
        n_methods = len(methods)

        box_width = 0.8 / n_methods
        offsets = np.linspace(-0.4 + box_width / 2, 0.4 - box_width / 2, n_methods)

        fig, ax = plt.subplots(figsize=(max(8, n_bins * 0.6), 5))

        legend_handles = []
        for m_idx, label in enumerate(methods):
            df = self.trajectories[self.trajectories['label'] == label]
            total_energy = df[energy_cols].sum(axis=1).values
            bin_indices = np.clip(np.digitize(df['realized_tariff'].values, edges) - 1, 0, n_bins - 1)

            color = label_colors[label]
            boxes = []
            positions = []
            for b in range(n_bins):
                data = total_energy[bin_indices == b]
                if len(data) > 0:
                    boxes.append(data)
                    positions.append(b + offsets[m_idx])

            if boxes:
                bp = ax.boxplot(boxes, positions=positions, widths=box_width,
                                patch_artist=True, manage_ticks=False,
                                medianprops=dict(color='k', linewidth=1.2),
                                whiskerprops=dict(linewidth=0.8),
                                capprops=dict(linewidth=0.8),
                                flierprops=dict(marker='.', markersize=5, alpha=0.8,
                                                markerfacecolor=color, markeredgecolor=color),
                                boxprops=dict(linewidth=0.8))
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7,
                               edgecolor='k', linewidth=0.8, label=label)
            )

        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Realized Tariff (EUR/MWh)')
        ax.set_ylabel('Total Energy (kWh)')
        ax.legend(handles=legend_handles)
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.18)
        return fig, ax

    def plot_cumulative_energy_by_price(self, methods=None):
        """
        Cumulative energy duration curve: one line per method label, all on one axes.

        Hours are sorted from cheapest to most expensive by realized tariff (same
        ordering for all methods — same experiment period, same prices).
        The y-axis shows cumulative total pump energy consumed up to each price rank.

        A price-aggressive schedule (LP) rises steeply at the left (most energy in
        cheapest hours), then flattens. A risk-averse schedule (RO) rises more linearly.

        A smaller subplot below shows the sorted realized tariff for price context.

        Args:
            methods:  list of labels to include; None = all labels
        """
        all_labels = self.trajectories['label'].unique().tolist()
        if methods is None:
            methods = all_labels

        label_to_group = {}
        for label in methods:
            row = self.trajectories[self.trajectories['label'] == label]
            if not row.empty:
                label_to_group[label] = row['group'].iloc[0]
        label_colors = build_label_colors(label_to_group)

        has_epanet = any(c.startswith('epanet_energy_') for c in self.trajectories.columns)
        if has_epanet:
            energy_cols = [c for c in self.trajectories.columns if c.startswith('epanet_energy_')]
        else:
            energy_cols = [c for c in self.trajectories.columns if c.startswith('opt_energy_')]

        # Sort order is shared across all methods (same realized tariff)
        ref_df = self.trajectories[self.trajectories['label'] == methods[0]].copy()
        sort_order = np.argsort(ref_df['realized_tariff'].values, kind='stable')
        sorted_tariff = ref_df['realized_tariff'].values[sort_order]
        x = np.arange(len(sort_order))

        fig = plt.figure(figsize=(9, 5))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.12, figure=fig)
        ax_energy = fig.add_subplot(gs[0])
        ax_tariff = fig.add_subplot(gs[1], sharex=ax_energy)

        for label in methods:
            df = self.trajectories[self.trajectories['label'] == label]
            if df.empty:
                raise ValueError(f"Label '{label}' not found. Available: {all_labels}")
            color = label_colors[label]
            total_energy = df[energy_cols].sum(axis=1).values[sort_order]
            ax_energy.plot(x, np.cumsum(total_energy), color=color, linewidth=1.2, label=label)

        ax_tariff.plot(x, sorted_tariff, color='k', linewidth=1.0)
        ax_tariff.fill_between(x, sorted_tariff, alpha=0.15, color='k')
        ax_tariff.set_xlabel('Hours (sorted by increasing price)')
        ax_tariff.set_ylabel('Tariff\n(€/MWh)')
        ax_tariff.grid(zorder=0)
        ax_tariff.set_axisbelow(True)
        ax_tariff.set_xlim(0, len(x) - 1)

        ax_energy.set_ylabel('Cumulative Energy (kWh)')
        ax_energy.grid(zorder=0)
        ax_energy.set_axisbelow(True)
        ax_energy.set_xlim(0, len(x) - 1)
        plt.setp(ax_energy.get_xticklabels(), visible=False)

        handles, labels_leg = ax_energy.get_legend_handles_labels()
        ncol = min(len(handles), 5)
        nrow = math.ceil(len(handles) / ncol)
        idx = list(range(len(handles))) + [-1] * (nrow * ncol - len(handles))
        order = [i for i in np.array(idx).reshape(nrow, ncol).T.flatten() if i >= 0]
        handles = [handles[i] for i in order]
        labels_leg = [labels_leg[i] for i in order]

        leg = fig.legend(handles, labels_leg, loc='upper left', bbox_to_anchor=(0.095, 0.96),
                         bbox_transform=fig.transFigure, ncol=ncol, fontsize=9, framealpha=0.9)
        for line in leg.get_lines():
            line.set_linewidth(2.2)

        fig.subplots_adjust(left=0.1, right=0.97, top=0.87, bottom=0.1)
        return fig, (ax_energy, ax_tariff)

    def plot_duration_curve(self, method):
        """
        Energy duration curve for a single method.

        Hours are sorted from cheapest (left) to most expensive (right) by
        realized tariff. The y-axis shows total EPANET energy consumed in each
        hour. A price-responsive scheduler produces a downward-sloping curve.
        The realized tariff is overlaid on a secondary y-axis for reference.

        Args:
            method: single label string
        """
        energy_cols = [c for c in self.trajectories.columns if c.startswith('epanet_energy_')]
        if not energy_cols:
            raise ValueError("No epanet_energy_* columns found in trajectories.")

        df = self.trajectories[self.trajectories['label'] == method].copy()
        df['total_energy'] = df[energy_cols].sum(axis=1)
        df = df.sort_values('realized_tariff').reset_index(drop=True)

        group = df['group'].iloc[0]
        color = mcolors.to_rgba(GROUP_STYLES[group]['color'])

        fig, ax1 = plt.subplots(figsize=(9, 4))

        x = np.arange(len(df))
        ax1.fill_between(x, df['total_energy'], alpha=0.5, color=color, step='mid')
        ax1.step(x, df['total_energy'], color=color, linewidth=1.0, where='mid', label=method)
        ax1.set_xlabel('Hours sorted by increasing price')
        ax1.set_ylabel('Total Energy (kWh)')
        ax1.set_xlim(0, len(df) - 1)
        ax1.grid(axis='y', alpha=0.4)
        ax1.set_axisbelow(True)

        ax2 = ax1.twinx()
        ax2.plot(x, df['realized_tariff'], color='grey', linestyle='--', alpha=0.7, label='Realized tariff')
        ax2.set_ylabel('Realized Tariff (EUR/MWh)', color='grey')
        ax2.tick_params(axis='y', labelcolor='grey')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.12)
        return fig, ax1, ax2


if __name__ == "__main__":
    # Example usage of the MPCGraphs class with sample data
    from network import LPNetwork
    import pandas as pd

    net = LPNetwork('data/multi_tank')
    traj = pd.read_csv('mpc_trajectories.csv')

    g = MPCGraphs(traj, net)
    g.plot_hydraulic_accuracy(methods=['LP', 'omega=2.0', 'alpha=[0.9, 0.9]'])
    g.plot_energy_accuracy(methods=['LP', 'omega=2.0', 'alpha=[0.9, 0.9]'])
    g.plot_hydraulic_scatter()

    group_styles = {
        'LP': {'color': '#333333', 'marker': 's'},
        'RO-Ellipsoid': {'color': '#105A7F', 'marker': 'o'},
        'CC-SP': {'color': '#FF9E00', 'marker': 'o'},
    }

    g.plot_por(pd.read_csv('mpc_summary.csv'), group_styles=group_styles)
    plt.show()
