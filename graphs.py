import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker as mtick


class OptGraphs:
    def __init__(self, opt_model):
        self.opt_model = opt_model
        self.x_fsp_value = opt_model.x_fsp_value
        self.x_vsp_value = opt_model.x_vsp_value
        self.t = self.opt_model.t

    def tank_volume(self, tank_id, plot_dem=False, plot_tariff=False, sample=None, ax=None):
        if sample.shape[1] > 1:
            alpha = 0.1
        else:
            alpha = 1.0

        vol = self.opt_model.network.get_tank_vol(t=self.opt_model.t, tank_id=tank_id,
                                                  x_fsp=self.x_fsp_value, x_vsp=self.x_vsp_value, sample=sample)
        if ax is None:
            fig, ax = plt.subplots()

        tank_data = self.opt_model.network.tanks.loc[tank_id]

        ax.plot(vol, c='C0', alpha=alpha)
        demands = self.opt_model.network.get_tank_demand(self.opt_model.t, tank_id)

        if plot_dem:
            ax.bar(range(len(demands)), demands, edgecolor='k', zorder=5)

        if plot_tariff:
            tariff = self.opt_model.network.tariffs['tariff'].values[:self.opt_model.t]
            ax.bar(range(len(tariff)), tariff, edgecolor='k', zorder=5)

        ax.hlines(tank_data['init_vol'], 0, self.t, colors='k', linestyles='--')
        ax.hlines(tank_data['min_vol'], 0, self.t, colors='red')
        ax.hlines(tank_data['max_vol'], 0, self.t, colors='red')
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

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 5))
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
            axes[-1].bar(range(self.t), self.opt_model.network.tariffs['tariff'].values[:self.t], edgecolor='k', alpha=0.6)
            axes[-1].set_ylabel('Tariff (€/MWh)')

        axes[-1].set_xlabel('Time (Hr)')


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

    def plot_por(self, summary, group_styles=None):
        fig, ax = plt.subplots(figsize=(6, 5))

        for group, grp_df in summary.groupby('group', sort=False):
            style = (group_styles or {}).get(group, {})
            color = style.get('color', None)
            marker = style.get('marker', 'o')
            ax.plot(grp_df['actual_cost'], grp_df['constraint_satisfaction_rate'],
                    color=color, marker=marker, label=group, markeredgecolor='k')

        ax.set_xlabel('Actual Cost')
        ax.set_ylabel('Constraint Satisfaction Rate')
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
