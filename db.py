import pandas as pd
import enlopy as el
import matplotlib.pyplot as plt

import utils


class DB:
    """ This holds the full data set for electricity tariffs and water demands
        The full data set is used for generating forecasts and building uncertainty sets
        If different ranges are needed for these purposes this will be handled in this class

        Other functionalities include visualization and analysis of the data
    """
    def __init__(self, demands_data_path, tariffs_data_path):
        self.demands_data_path = demands_data_path
        self.tariffs_data_path = tariffs_data_path

        self.demands = None
        self.tariffs = None
        self.data = None

        self.load_data()

    def load_data(self, start_date=None, end_date=None):
        demands = pd.read_csv(self.demands_data_path, index_col=0)
        tariffs = pd.read_csv(self.tariffs_data_path, index_col=0)[["tariff"]]

        tariffs.index = pd.to_datetime(tariffs.index, utc=True)
        demands.index = pd.to_datetime(demands.index, utc=True)

        if start_date is not None:
            tariffs = tariffs.loc[tariffs.index >= start_date]
            demands = demands.loc[demands.index >= start_date]
        if end_date is not None:
            tariffs = tariffs.loc[tariffs.index <= end_date]
            demands = demands.loc[demands.index <= end_date]

        common_index = tariffs.index.intersection(demands.index)

        self.tariffs = tariffs.loc[common_index].sort_index()
        self.demands = demands.loc[common_index].sort_index()

        elec_demand_data = pd.concat([self.tariffs, self.demands], axis=1)
        elec_demand_data.index = pd.to_datetime(elec_demand_data.index, dayfirst=True)
        self.data = elec_demand_data

        # Make sure no missing data in the tariff data set
        self.data = utils.data_completion(self.data)
        elec_demand_data = utils.data_completion(elec_demand_data)
        return elec_demand_data

    def visualize_data(self):
        data = self.data.copy()
        data['day'] = data.index.day
        data['hour'] = data.index.hour
        data['day_of_year'] = data.index.dayofyear
        data['weekday'] = data.index.weekday
        data['month'] = data.index.month

        fig, ax = plt.subplots(len(self.data.columns), 1, sharex=True)
        axes = ax.ravel()
        for i, col in enumerate(self.data.columns):
            el.plot_heatmap(data.iloc[:, i], x='day_of_year', y='hour', aggfunc='mean', cmap="Blues", edgecolors='k',
                            ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_ylabel('')
            if i != len(self.data.columns) - 1:
                axes[i].set_xlabel('')

        fig.text(0.02, 0.5, 'Hour', va='center', rotation='vertical')

        fig, ax = plt.subplots(len(self.data.columns), 1, sharex=True)
        axes = ax.ravel()
        for i, col in enumerate(self.data.columns):
            el.plot_heatmap(data.iloc[:, i], x='hour', y='weekday', aggfunc='mean', cmap="Blues", edgecolors='k',
                            ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_ylabel('')
            if i != len(self.data.columns) - 1:
                axes[i].set_xlabel('')

        fig.text(0.02, 0.5, 'Weekday', va='center', rotation='vertical')

        fig, ax = plt.subplots(len(self.data.columns), 1, sharex=True)
        axes = ax.ravel()
        for i, col in enumerate(self.data.columns):
            el.plot_percentiles(data.iloc[:, i], x='hour', zz='day_of_year',
                                perc_list=[[1, 99], [25, 75], 50], ax=axes[i])

    def hourly_distribution(self):
        import textwrap

        data = self.data.copy()
        data['day'] = data.index.day
        data['hour'] = data.index.hour
        data['day_of_year'] = data.index.dayofyear
        data['weekday'] = data.index.weekday
        data['month'] = data.index.month
        columns = data.columns.difference(['day', 'hour', 'day_of_year', 'weekday', 'month'])

        fig_box, axes_box = plt.subplots(nrows=len(columns), sharex=True, figsize=(8, 6))
        fig_lines, axes_lines = plt.subplots(nrows=len(columns), sharex=True, sharey=False, figsize=(8, 6))
        fig_hist, axes_hist = plt.subplots(nrows=len(columns), sharex=True, figsize=(8, 6))

        for i, col in enumerate(data[columns].columns):
            temp = data[[col]]
            pivot_temp = temp.pivot_table(values=col, index=temp.index.date, columns=temp.index.hour)
            axes_box[i].boxplot(pivot_temp.values, positions=range(24))
            axes_box[i].set_ylabel(col)

            axes_lines[i].plot(pivot_temp.values.T, c='C0', alpha=0.1)
            axes_lines[i].set_ylabel(col)
            axes_lines[i].grid()

            axes_hist[i].hist(data[col], bins=25, edgecolor='k', alpha=0.5)
            axes_hist[i].set_ylabel(col)
            if col == "tariff":
                axes_hist[i].set_xlabel("Tariff (€/MWh)")
            else:
                axes_hist[i].set_xlabel("Demand ($m^3/hr$)")

        fig_box.align_ylabels()
        fig_box.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.9, hspace=0.15)
        fig_box.text(0.5, 0.02, 'Hour of the day', ha='center')

        fig_lines.align_ylabels()
        fig_lines.subplots_adjust(bottom=0.1, top=0.98, left=0.1, right=0.9, hspace=0.2)
        fig_lines.text(0.5, 0.02, 'Hour of the day', ha='center')

        fig_hist.align_ylabels()
        fig_hist.subplots_adjust(bottom=0.1, top=0.98, left=0.12, right=0.95, hspace=0.25)


if __name__ == "__main__":
    # exploratory data anlysis
    db = DB('data/multi_tank/demands2022.csv', 'data/multi_tank/tariffs2022.csv')
    db.visualize_data()
    db.hourly_distribution()
    plt.show()