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
    'LP':            {'color': 'k',        'cmap': 'Greys',    'marker': 'x'},
    'RO-Ellipsoid':  {'color': 'C0',  'cmap': 'Blues',    'marker': 'o'},
    'RO-Budget':     {'color': '#6a0dad',  'cmap': 'Purples',  'marker': 'D'},
    'CC-SP':         {'color': '#d13b2c',  'cmap': 'Reds',     'marker': 's'},
    'CC-SP-relaxed': {'color': 'C1',  'cmap': 'Oranges',  'marker': '^'},
}


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

    def plot_tank_trajectories(self, methods=None, show_opt=False, figsize=None):
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
            figsize:  override default figure size
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
        fig = plt.figure(figsize=figsize or (10, 2 * n_tanks + 1))
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
                time = pd.to_datetime(df['time'], dayfirst=True)
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
        ref_time = pd.to_datetime(ref_df['time'], dayfirst=True)
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

    def plot_por(self, summary, violations_metric="violated_vol"):
        """
        Plot cost vs. violations (POR curve) for all method groups.

        summary:           DataFrame with 'group', 'actual_cost', and the
                           violations_metric column.
        violations_metric: 'violated_vol' or 'constraint_satisfaction_rate'

        Colors and markers are taken from GROUP_STYLES keyed by group name.
        All groups present in the summary must be defined in GROUP_STYLES.
        """
        fig, ax = plt.subplots(figsize=(6, 5))

        for group, grp_df in summary.groupby('group', sort=False):
            style = GROUP_STYLES[group]
            ax.scatter(grp_df['actual_cost'], grp_df[violations_metric],
                    color=style['color'], marker=style['marker'], s=40,
                    label=group, edgecolor='k')

        ax.set_xlabel('Actual Cost (€)')
        ax.set_ylabel({"violated_vol": "Violated Volume ($m^3$)",
                       "constraint_satisfaction_rate": 'Constraint Satisfaction Rate'}[violations_metric])
        ax.legend()
        ax.grid()
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.1)
        return fig, ax


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
