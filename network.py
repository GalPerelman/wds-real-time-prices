import copy
import math
import os

import numpy as np
import pandas as pd
import wntr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def write_demand_patterns(csv_path, inp_path, output_path, pattern_timestep=3600):
    """
    Read a CSV with one column per junction (absolute demand values in CMH),
    write each column as a pattern assigned to that junction with base demand
    of 1 CMH (so the pattern multipliers equal the CMH demand values), and
    export the resulting network to output_path.

    example usage:
    write_demand_patterns(csv_path="data/two_mains/demand_patterns.csv",
    #                     inp_path="data/two_mains/two_mains.inp",
    #                     output_path="data/two_mains/two_mains_dem.inp")
    """
    df = pd.read_csv(csv_path)
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.time.pattern_timestep = pattern_timestep
    wn.options.time.duration = len(df) * pattern_timestep

    base_value = 1.0 / 3600.0
    for junction_id in df.columns:
        junction_id = str(junction_id)
        junction = wn.get_node(junction_id)

        pattern_name = f"{junction_id}"
        multipliers = df[junction_id].tolist()
        if pattern_name in wn.pattern_name_list:
            wn.get_pattern(pattern_name).multipliers = multipliers
        else:
            wn.add_pattern(pattern_name, multipliers)

        junction.demand_timeseries_list.clear()
        junction.add_demand(base_value, pattern_name)

    wntr.network.io.write_inpfile(wn, output_path)
    return wn


class HydraulicOperatingPoints:
    """
    helper to analyze hydraulics and extract pump operating points
    """
    def __init__(self, inp_path):
        self.inp_path = inp_path
        self.wn = wntr.network.WaterNetworkModel(self.inp_path)

        self.all_elements = {'pipe': self.wn.pipe_name_list,
                             'pump': self.wn.pump_name_list,
                             'valve': self.wn.valve_name_list,
                             'junction': self.wn.junction_name_list,
                             'tank': self.wn.tank_name_list,
                             'reservoir': self.wn.reservoir_name_list
                             }
        self.sim_results = self.run_sim()

    def run_sim(self):
        sim = wntr.sim.EpanetSimulator(self.wn)
        return sim.run_sim()

    def _pump_efficiency(self, pump_name, flow):
        """
        Interpolate pump efficiency (fraction) from its EFFICIENCY curve at the given flow values (m3/s).
        Flow <= 0 maps to efficiency 0 without interpolation.
        Flow outside the curve's range is clamped to the nearest endpoint efficiency. If the pump has
        no EFFICIENCY curve, use EPANET's constant Global Efficiency instead (same fallback EPANET
        itself applies when a pump has no assigned curve).
        """
        flow = np.asarray(flow, dtype=float)
        eff = np.zeros_like(flow)
        mask = flow > 0
        curve = self.wn.get_link(pump_name).efficiency_curve
        if curve is None:
            eff[mask] = self.wn.options.energy.global_efficiency / 100.0
        else:
            points = sorted(curve.points)
            xs = np.array([p[0] for p in points])
            ys = np.array([p[1] for p in points]) / 100.0
            eff[mask] = np.interp(flow[mask], xs, ys)
        return eff

    def plot_pumps_operational_points(self, remove_controls=True, all_pumps_on=True):
        wn = copy.deepcopy(self.wn)
        if remove_controls:
            for ctrl_name in wn.control_name_list:
                wn.remove_control(ctrl_name)
        if all_pumps_on:
            for pump_name, pump in wn.pumps():
                pump.initial_status = 1
                pump._status = 1

        sim = wntr.sim.EpanetSimulator(wn)
        temp_results = sim.run_sim()

        q = temp_results.link["flowrate"].loc[:, self.all_elements["pump"]].round(4)
        h = - temp_results.link["headloss"].loc[:, self.all_elements["pump"]].round(4)

        power = pd.DataFrame(0.0, index=q.index, columns=q.columns)
        for pump_name in q.columns:
            eff = self._pump_efficiency(pump_name, q[pump_name].values)
            mask = eff > 0
            # kW, efficiency from pump curve
            power.loc[mask, pump_name] = (9810 * q.loc[mask, pump_name] * h.loc[mask, pump_name] / (eff[mask] * 1000))

        n = len(q.columns)
        ncols = max(1, int(math.ceil(math.sqrt(n))))
        nrows = max(1, int(math.ceil(n / ncols)))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(9, 5))
        axes = np.atleast_2d(axes).ravel()

        for i, col in enumerate(q.columns):
            axes[i].scatter(q.iloc[:, i] * 3600, h.iloc[:, i], zorder=5, alpha=0.5)
            axes[i].set_title(col)
            axes[i].grid(zorder=0)

        fig.supylabel('Head (m)')
        fig.supxlabel('Flow (CMH)')
        fig.subplots_adjust(left=0.1, bottom=0.15)

        fig, axes = plt.subplots(nrows=3, figsize=(8, 5))

        q = q.where(q > 0)
        power = power.where(power > 0)

        axes[0].boxplot([col.dropna().values * 3600 for _, col in q.items()], labels=q.columns)
        axes[0].set_ylabel('Flow (CMH)')
        axes[0].set_xlabel('Pump')

        axes[1].boxplot([col.dropna().values for _, col in power.items()], labels=power.columns)
        axes[1].set_ylabel('Power (kW)')
        axes[1].set_xlabel('Pump')
        print(pd.DataFrame(q).describe())
        print(pd.DataFrame(power).describe())

        df = pd.DataFrame(q * 3600).describe()
        df = df.round(1)
        pd.plotting.table(axes[2], df, loc='center')
        axes[2].axis('off')
        fig.subplots_adjust(left=0.1, bottom=0.08, hspace=0.5, top=0.96, right=0.95)

        stations = {}
        for pump_name in self.all_elements['pump']:
            pump = self.wn.get_link(pump_name)
            key = (pump.start_node_name, pump.end_node_name)
            stations.setdefault(key, []).append(pump_name)

        n_st = len(stations)
        ncols = max(1, int(math.sqrt(n_st))) + 1
        nrows = math.ceil(n_st / ncols)
        n_axes = nrows * ncols
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(ncols * 3, nrows * 1.8))
        axes = axes.ravel()

        operating_points = {}
        for ax, ((suction, discharge), pump_names) in zip(axes, stations.items()):
            station_q = q[pump_names].sum(axis=1) * 3600   # CMH; NaN (pump off) treated as 0
            station_p = power[pump_names].sum(axis=1)       # kW

            mask = station_q > 0
            flow_vals = station_q[mask].values
            power_vals = station_p[mask].values
            ax.scatter(flow_vals, power_vals, alpha=0.5, s=10)

            station_name = pump_names[0][:3].upper()
            n_clusters = min(len(pump_names), len(flow_vals))
            if n_clusters > 0:
                x = np.column_stack([flow_vals, power_vals])
                scaler = StandardScaler()
                x_scaled = scaler.fit_transform(x)
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(x_scaled)
                centers = scaler.inverse_transform(kmeans.cluster_centers_)
                centers = centers[centers[:, 0].argsort()]
                ax.scatter(centers[:, 0], centers[:, 1], color='red', marker='o', s=50,
                           edgecolors='k', linewidths=0.5, zorder=5, label='operating points')
                ax.legend(fontsize=7)
                operating_points[station_name] = pd.DataFrame(centers, columns=['flow_cmh', 'power_kw'])

            ax.set_xlabel('Flow (CMH)', fontsize=10)
            ax.set_ylabel('Power (kW)', fontsize=10)
            ax.set_title(station_name, fontsize=10)
            ax.grid()

        for i in range(len(stations), n_axes):
            axes[i].axis('off')

        fig.suptitle('Pump stations operating points K-Means Clustering', fontsize=10)
        fig.tight_layout()
        all_points = pd.concat(operating_points, names=['station', 'combo']).reset_index()
        all_points['combo'] += 1
        all_points = all_points[['station', 'combo', 'flow_cmh', 'power_kw']].round(1)

        print('\nAll stations operating points:')
        print(all_points.to_string(index=False))

        return operating_points


class LPNetwork:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        tanks_dtypes = {"min_vol": float, "max_vol": float, "init_vol": float, "final_vol": float}

        self.fsp = pd.read_csv(os.path.join(self.data_dir, 'fsp.csv'), index_col=0)
        self.vsp = pd.read_csv(os.path.join(self.data_dir, 'vsp.csv'), index_col=0)
        self.tanks = pd.read_csv(os.path.join(self.data_dir, 'tanks.csv'), index_col=0, dtype=tanks_dtypes)

        # placeholder for forecast values
        self.demands = None
        self.tariffs = None

        self.n_fsp = len(self.fsp)
        self.n_stations = self.fsp["name"].nunique()
        self.n_vsp = len(self.vsp)
        self.n_tanks = len(self.tanks)

    def get_tank_demand(self, t, tank_id):
        tank_consumer_id = self.tanks.loc[tank_id, "demand"]
        return self.demands.iloc[:t].loc[:, tank_consumer_id]

    def get_tank_vol(self, t, tank_id, x_fsp, x_vsp, sample=None, include_init=False):
        mat = np.tril(np.ones((t, t)))
        if sample is None:
            tank_consumer_id = self.tanks.loc[tank_id, "demand"]
            cumm_tank_consumption = self.demands.iloc[:t].loc[:, tank_consumer_id].values.cumsum().reshape(-1, 1)
        else:
            # Handle both 1D (MPC with wait) and 2D (Monte Carlo samples) inputs
            cumm_tank_consumption = sample.cumsum(axis=0)
            if cumm_tank_consumption.ndim == 1:
                cumm_tank_consumption = cumm_tank_consumption.reshape(-1, 1)

        init_vol = self.tanks.loc[tank_id, "init_vol"]
        lhs = init_vol
        for i, (fsp_name, row_fsp) in enumerate(self.fsp.iterrows()):
            if row_fsp['in'] == tank_id:
                lhs += row_fsp['flow'] * mat @ x_fsp[:, i]
            elif row_fsp['out'] == tank_id:
                lhs += -1 * row_fsp['flow'] * mat @ x_fsp[:, i]
            else:
                continue

        for i, (vsp_idx, row_vsp) in enumerate(self.vsp.iterrows()):
            if row_vsp['in'] == tank_id:
                lhs += mat @ x_vsp[:, i]
            elif row_vsp['out'] == tank_id:
                lhs += -1 * mat @ x_vsp[:, i]
            else:
                continue

        vol = lhs.reshape(-1, 1) - cumm_tank_consumption
        if include_init:
            n_cols = vol.shape[1]
            init_row = np.full((1, n_cols), init_vol)
            vol = np.vstack([init_row, vol])
        return vol

    @staticmethod
    def level_to_vol(diameter, level):
        return level * np.pi * (diameter ** 2) / 4


if __name__ == "__main__":
    net = HydraulicOperatingPoints("data/price_ostfeld/price_ostfeld.inp")
    net.plot_pumps_operational_points(remove_controls=False)

    net = HydraulicOperatingPoints("data/two_mains/two_mains.inp")
    net.plot_pumps_operational_points(remove_controls=False)
    plt.show()