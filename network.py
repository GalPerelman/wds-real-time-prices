import copy
import math
import os

import numpy as np
import pandas as pd
import wntr
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

import utils


class NLPNetwork:
    """
    Not in use for now
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

    def generate_bounds(self):
        wn = copy.deepcopy(self.wn)
        for ctrl_name in wn.control_name_list:
            wn.remove_control(ctrl_name)
        for pump_name, pump in wn.pumps():
            pump.initial_status = 1

        sim = wntr.sim.EpanetSimulator(wn)
        temp_results = sim.run_sim()

        links_min_flow = temp_results.link["flowrate"].min(axis=0) * 3600
        links_max_flow = temp_results.link["flowrate"].max(axis=0) * 3600
        return links_min_flow, links_max_flow

    def get_pump_head_curve(self, pump_id):
        pump = self.wn.get_link(pump_id)
        curve_points = pump.get_pump_curve().points
        if len(curve_points) == 1:
            q, h = curve_points[0][0] * 3600, curve_points[0][1]
            xdata = [0, q, 2 * q]
            ydata = [(4/3) * h, h, 0]
        else:
            xdata = [point[0] * 3600 for point in curve_points]
            ydata = [point[1] for point in curve_points]

        def pump_poly(x, a, b):
            return a - b * (x ** 2)

        popt, pcov = curve_fit(pump_poly, xdata, ydata)
        a, b = popt
        return a, b

    def piecewise_linear(self, n_pl):
        # pipes
        pipes_pl = {}
        length = self.wn.query_link_attribute(attribute="length")
        diameter = self.wn.query_link_attribute(attribute="diameter")
        c_hw = self.wn.query_link_attribute(attribute="roughness")

        links_min_flow, links_max_flow = self.generate_bounds()
        pipes = pd.concat([length, diameter, c_hw, links_min_flow, links_max_flow], axis=1, join="inner")
        pipes.columns = ["length", "diameter", "roughness", "min_flow", "max_flow"]
        pipes["r"] = ((1.1311 * (10 ** 9)) * ((1 / pipes["roughness"]) ** 1.852)
                      * ((1 / (pipes["diameter"] * 1000)) ** 4.87) * pipes["length"])

        for pipe_id, row in pipes.iterrows():
            if self.wn.get_link(pipe_id).cv:
                continue
            x = np.linspace(row['min_flow'], row["max_flow"], n_pl)
            h = row["r"] * x * (np.abs(x)) ** 0.852
            head_points = UnivariateSpline(x, h, k=1, s=0)
            pl = {}
            for i in range(len(x)):
                pl[i] = {'flow': x[i], 'head': float(head_points(x[i]))}
            pipes_pl[pipe_id] = pl

        # pumps
        pumps_pl = {}
        for pump_id, pump in self.wn.pumps():
            a, b = self.get_pump_head_curve(pump_id)
            x = np.linspace(links_min_flow.loc[pump_id], links_max_flow.loc[pump_id], n_pl)
            h = a - b * (x ** 2)

            head_points = UnivariateSpline(x, h, k=1, s=0)
            pl = {}
            for i in range(len(x)):
                pl[i] = {'flow': x[i], 'head': float(head_points(x[i]))}
            pumps_pl[pump_id] = pl

    def plot_sim_results(self, element_type, param):
        if element_type in ["pipe", "pump", "valve"]:
            y = self.sim_results.link[param].loc[:, self.all_elements[element_type][:15]]
        if element_type in ["junction", "tank", "reservoir"]:
            y = self.sim_results.node[param].loc[:, self.all_elements[element_type]]

        n = len(y.columns)
        ncols = max(1, int(math.ceil(math.sqrt(n))))
        nrows = max(1, int(math.ceil(n / ncols)))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(8, 4))
        axes = np.atleast_2d(axes).ravel()

        for i, col in enumerate(y.columns):
            axes[i].plot(y.iloc[:, i])
            axes[i].set_title(col)
            axes[i].grid()

    def plot_pumps_operational_points(self):
        # deactivate all controls
        wn = copy.deepcopy(self.wn)
        # for ctrl_name in wn.control_name_list:
        #     wn.remove_control(ctrl_name)
        # for pump_name, pump in wn.pumps():
        #     pump.initial_status = 1
        #     pump._status = 1

        sim = wntr.sim.EpanetSimulator(wn)
        temp_results = sim.run_sim()

        q = temp_results.link["flowrate"].loc[:, self.all_elements["pump"]]
        h = - temp_results.link["headloss"].loc[:, self.all_elements["pump"]]
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

        fig, axes = plt.subplots(nrows=2)
        q = q[q > 0]
        axes[0].boxplot(q * 3600)
        axes[0].set_ylabel('Flow (CMH)')
        axes[0].set_xlabel('Pump')
        print(pd.DataFrame(q * 3600).describe())

        df = pd.DataFrame(q * 3600).describe()
        df = df.round(1)
        pd.plotting.table(axes[1], df, loc='center')
        axes[1].axis('off')
        fig.subplots_adjust(left=0.1, bottom=0.1, hspace=0.3, top=0.95, right=0.95)


class LPNetwork:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.fsp = pd.read_csv(os.path.join(self.data_dir, 'fsp.csv'), index_col=0)
        self.vsp = pd.read_csv(os.path.join(self.data_dir, 'vsp.csv'), index_col=0)
        self.tanks = pd.read_csv(os.path.join(self.data_dir, 'tanks.csv'), index_col=0)
        self.demands = pd.read_csv(os.path.join(self.data_dir, 'demands.csv'), index_col=0)
        self.tariffs = pd.read_csv(os.path.join(self.data_dir, 'tariffs.csv'), index_col=0)

        self.n_fsp = len(self.fsp)
        self.n_stations = self.fsp["name"].nunique()
        self.n_vsp = len(self.vsp)
        self.n_tanks = len(self.tanks)

        # Make sure nomissing data in the tariff data set
        self.tariffs = utils.data_completion(self.tariffs[["tariff"]])

    def get_tank_demand(self, t, tank_id):
        tank_consumer_id = self.tanks.loc[tank_id, "demand"]
        return self.demands.iloc[:t].loc[:, tank_consumer_id]

    def get_tank_vol(self, t, tank_id, x_fsp, x_vsp, sample=None):
        mat = np.tril(np.ones((t, t)))
        if sample is None:
            tank_consumer_id = self.tanks.loc[tank_id, "demand"]
            cumm_tank_consumption = self.demands.iloc[:t].loc[:, tank_consumer_id].values.cumsum()
        else:
            cumm_tank_consumption = sample.cumsum(axis=0)

        lhs = self.tanks.loc[tank_id, "init_vol"]
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
        return vol

    @staticmethod
    def level_to_vol(diameter, level):
        return level * np.pi * (diameter ** 2) / 4


if __name__ == "__main__":
    net = NLPNetwork("data/multi_tank.inp")
    net.piecewise_linear(n_pl=4)
    net.plot_pumps_operational_points()
    plt.show()