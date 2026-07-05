import pandas as pd
import enlopy as el
import matplotlib.pyplot as plt

import utils


class DataManager:
    """ Single source of truth for electricity tariffs and water demands data.

    Provides separate methods for:
    - Loading full dataset
    - Getting data for uncertainty estimation (typically narrower date range)
    - Getting data for forecaster training (typically larger date range)

    Also includes visualization and analysis methods.
    """
    def __init__(self, demands_data_path, tariffs_data_path, prices_col):
        self.demands_data_path = demands_data_path
        self.tariffs_data_path = tariffs_data_path
        self.prices_col = prices_col

        self.demands = None
        self.tariffs = None
        self.data = None
        self.uncertainty_training_data = None  # raw data for building the uncertainty set
        self.forecast_training_data = None  # raw data for training the forecaster

        self.load_data()

    def load_data(self, start_date=None, end_date=None):
        demands = pd.read_csv(self.demands_data_path, index_col=0)
        tariffs = pd.read_csv(self.tariffs_data_path, index_col=0)[[self.prices_col]]

        # Validate tz-awareness on raw strings before conversion (utc=True would silently accept tz-naive)
        if pd.to_datetime(tariffs.index[0]).tzinfo is None:
            raise ValueError(f"Timestamps in {self.tariffs_data_path} are tz-naive; tz-aware timestamps required.")
        if pd.to_datetime(demands.index[0]).tzinfo is None:
            raise ValueError(f"Timestamps in {self.demands_data_path} are tz-naive; tz-aware timestamps required.")

        # Convert to UTC and strip timezone: tz-naive UTC index, DST resolved by the conversion.
        tariffs.index = pd.to_datetime(tariffs.index, utc=True).tz_convert(None)
        demands.index = pd.to_datetime(demands.index, utc=True).tz_convert(None)

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
        self.data = elec_demand_data

        # Make sure no missing data in the tariff data set
        self.data = utils.data_completion(self.data)
        elec_demand_data = utils.data_completion(elec_demand_data)
        return elec_demand_data

    def set_uncertainty_training_window(self, start_date, end_date):
        """Set the date range for uncertainty estimation training data.

        Args:
            start_date: Start of the window (e.g., '2022-06-01')
            end_date: End of the window (e.g., '2022-07-31')
        """
        data = self.data.copy()
        data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
        self.uncertainty_training_data = data
        return data

    def set_forecast_training_window(self, start_date, end_date):
        """Set the date range for forecaster training data.

        Args:
            start_date: Start of the window (e.g., '2022-01-01')
            end_date: End of the window (e.g., '2022-12-31')
        """
        data = self.data.copy()
        data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
        self.forecast_training_data = data
        return data

    def visualize_data(self):
        data = self.data.copy()
        data['day'] = data.index.day
        data['hour'] = data.index.hour
        data['day_of_year'] = data.index.dayofyear
        data['weekday'] = data.index.weekday
        data['month'] = data.index.month

        fig, axes = plt.subplots(len(self.data.columns), 1, sharex=True)
        for i, col in enumerate(self.data.columns):
            axes[i].plot(data[col])
            axes[i].set_title(col)
            if col == self.prices_col:
                axes[i].set_ylabel("Tariff (€/MWh)")
            else:
                axes[i].set_ylabel("Demand ($m^3/hr$)")
            axes[i].grid()
            axes[-1].set_xlabel("Time")
        plt.subplots_adjust(hspace=0.32, top=0.95, bottom=0.1, right=0.92)
        fig.autofmt_xdate()

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
        data = self.data.copy()
        data['day'] = data.index.day
        data['hour'] = data.index.hour
        data['day_of_year'] = data.index.dayofyear
        data['weekday'] = data.index.weekday
        data['month'] = data.index.month
        columns = data.columns.difference(['day', 'hour', 'day_of_year', 'weekday', 'month'])

        mask = data.isna().any(axis=1)
        rows_with_nan = data.loc[mask]

        fig_box, axes_box = plt.subplots(nrows=len(columns), sharex=True, figsize=(8, 6))
        fig_lines, axes_lines = plt.subplots(nrows=len(columns), sharex=True, sharey=False, figsize=(8, 6))
        fig_hist, axes_hist = plt.subplots(nrows=len(columns), sharex=True, figsize=(8, 6))

        for i, col in enumerate(data[columns].columns):
            temp = data[[col]]
            pivot_temp = temp.pivot_table(values=col, index=temp.index.date, columns=temp.index.hour)

            axes_box[i].boxplot(pivot_temp.values, positions=range(24))

            axes_lines[i].plot(pivot_temp.values.T, c='C0', alpha=0.1)
            axes_lines[i].grid()

            axes_hist[i].hist(data[col], bins=25, edgecolor='k', alpha=0.5)
            axes_hist[i].set_ylabel('Count')
            if col == self.prices_col:
                axes_hist[i].set_xlabel("Tariff (€/MWh)")
                axes_lines[i].set_ylabel("Tariff (€/MWh)")
                axes_box[i].set_ylabel("Tariff (€/MWh)")
            else:
                axes_hist[i].set_xlabel("Demand ($m^3/hr$)")
                axes_lines[i].set_ylabel("Demand ($m^3/hr$)")
                axes_box[i].set_ylabel("Demand ($m^3/hr$)")

        fig_box.align_ylabels()
        fig_box.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.9, hspace=0.15)
        fig_box.text(0.5, 0.02, 'Hour of the day', ha='center')

        fig_lines.align_ylabels()
        fig_lines.subplots_adjust(bottom=0.1, top=0.98, left=0.1, right=0.9, hspace=0.2)
        fig_lines.text(0.5, 0.02, 'Hour of the day', ha='center')

        fig_hist.align_ylabels()
        fig_hist.subplots_adjust(bottom=0.1, top=0.98, left=0.12, right=0.95, hspace=0.25)


if __name__ == "__main__":
    # exploratory data analysis
    dm = DataManager('data/multi_tank/demands.csv', 'data/multi_tank/tariffs.csv')
    dm.visualize_data()
    dm.hourly_distribution()
    plt.show()