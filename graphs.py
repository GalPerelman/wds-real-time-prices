import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick


class OptGraphs:
    def __init__(self, opt_model):
        self.opt_model = opt_model
        self.x_fsp = opt_model.x_fsp
        self.x_vsp = opt_model.x_vsp
        self.t = self.opt_model.t

    def tank_volume(self, tank_id, plot_dem=False, plot_tariff=False, sample=None, ax=None):
        vol = self.opt_model.network.get_tank_vol(t=self.opt_model.t, tank_id=tank_id,
                                                  x_fsp=self.x_fsp, x_vsp=self.x_vsp, sample=sample)
        if ax is None:
            fig, ax = plt.subplots()

        tank_data = self.opt_model.network.tanks.loc[tank_id]

        ax.plot(vol, c='C0')
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

    def all_tanks(self, plot_dem=False, plot_tariff=False, sample=None):
        ncols = max(1, int(math.ceil(math.sqrt(self.opt_model.network.n_tanks))))
        nrows = max(1, int(math.ceil(self.opt_model.network.n_tanks / ncols)))

        fig, axes = plt.subplots(nrows=ncols, ncols=nrows, figsize=(10, 6))
        axes = axes.ravel()
        for i, (tank_id, tank_data) in enumerate(self.opt_model.network.tanks.iterrows()):
            if tank_id == 9:
                continue

            if sample is not None:
                dem_sample = sample[self.t + i * self.t: self.t + i * self.t + self.t:, :]
            else:
                dem_sample = None

            self.tank_volume(tank_id, ax=axes[i], plot_dem=plot_dem, plot_tariff=plot_tariff, sample=dem_sample)
            axes[i].set_title(tank_id)


def plot_mat(mat, norm=False, t=24):
    fig = plt.figure()
    if norm:
        mat = (mat - mat.min()) / (mat.max() - mat.min())

    mat_norm = max(abs(mat.min()), abs(mat.max()))
    im = plt.imshow(mat, cmap="bwr")
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