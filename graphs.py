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