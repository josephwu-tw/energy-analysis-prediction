import pandas as pd
import datetime as dt
import seaborn as sns
import calendar
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_processing import get_df


# setting parameters

dataframe = get_df('comb')
period = 'month' # ['all', 'month', 'week', 'date']
y_col = 'Total'
# y_col = list(['df.columns[1:-1]']) # enable if multiple = True

capacity_max = 600
battery_output_max = 400
battery_charge_limit = capacity_max - battery_output_max

start_date = dataframe['Time'].iloc[0].date()
end_date = dataframe['Time'].iloc[-1].date()

# function

def plot(dataframe = dataframe, period = period, y_col = y_col, multiple = False, capacity = False, charge_limit = False, output_limit = False):
  if period not in ['all', 'month', 'week', 'date']:
          raise ValueError("Invalid value for 'period'. Expected 'all', 'month', 'week' or 'date'.")

  if period == 'all':
      title_name = 'Power Consumption per 15 min'
      xaxis_locator = mdates.MonthLocator()
      date_formatter = '%Y-%m'

  else:
      print(f"Strat Date : {start_date.strftime('%Y-%m-%d')}")
      print(f"End   Date : {end_date.strftime('%Y-%m-%d')}")

      if period in ['month', 'week']:
          year_idx = int(input(f"Enter year ({start_date.year}-{end_date.year}) : "))

          if year_idx > end_date.year or year_idx < start_date.year:
              raise ValueError(f"Invalid year. Expected range from {start_date.year} to {end_date.year}.")

          if period == 'month':

              month_start = start_date.month if year_idx == start_date.year else 1
              month_end = end_date.month if year_idx == end_date.year else 12

              month_idx = int(input(f"Enter month index ({month_start}-{month_end}) : "))

              if month_idx > month_end or month_idx < month_start:
                  raise ValueError(f"Invalid value for 'month index'. Expected range from {month_start} to {month_end}.")

              dataframe = dataframe[dataframe['Time'].apply(lambda x: (x.year == year_idx and x.month == month_idx))]

              title_name = f'Power Consumption per 15 min in {year_idx} {calendar.month_abbr[month_idx]}'

          elif period == 'week':

              week_start = start_date.isocalendar()[1] if year_idx == start_date.year else 1
              week_end = end_date.isocalendar()[1] if year_idx == end_date.year else 53


              week_idx = int(input(f"Enter week index ({week_start}-{week_end}) : "))

              if week_idx > week_end or week_idx < week_start:
                  raise ValueError(f"Invalid value for 'week index'. Expected range from {week_start} to {week_end}.")

              dataframe = dataframe[dataframe['Time'].apply(lambda x: x.isocalendar()[:2] == (year_idx, week_idx))]

              title_name = f'Power Consumption per 15 min in {year_idx} WK{week_idx}'

          xaxis_locator = mdates.DayLocator()
          date_formatter = '%m-%d'

      elif period == 'date':

          date = dt.datetime.strptime(input("Enter date (ex: 2024-01-01) : "), '%Y-%m-%d')

          if date.date() > end_date or date.date() < start_date:
              raise ValueError(f"Invalid date. Expected date between {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")

          dataframe = dataframe[(dataframe['Time'] <= date + dt.timedelta(days=1)) & (dataframe['Time'] > date)]

          title_name = f"Power Consumption per 15 min in {date.strftime('%Y-%m-%d')}"
          date_formatter = '%H-%M'

  palette = list(sns.palettes.mpl_palette('Dark2'))
  x = dataframe['Time']
  fig, ax = plt.subplots(figsize = (20,10))

  if multiple:
      if type(y_col) is not list:
          raise ValueError("Invalid value for 'y_col' when 'multiple' is True. Expected list type input.")

      for col in y_col:
          y = dataframe[col]
          ax.plot(x, y)

      ax.set(xlabel = 'Time', ylabel = 'kWh',
             title = title_name)
      ax.legend(y_col)

  else:
      if type(y_col) is not str:
          raise ValueError("Invalid value for 'y_col' when 'multiple' is False. Expected string type input.")

      y = dataframe[y_col]
      ax.plot(x, y)

      ax.set(xlabel = 'Time', ylabel = 'kWh',
             title = y_col + ' ' + title_name)


  if period in ['total', 'week']:
      ax.set_xticks(x)
      ax.xaxis.set_major_locator(xaxis_locator)
  ax.xaxis.set_major_formatter(mdates.DateFormatter(date_formatter))

  if capacity:
        plt.axhline(y = capacity_max//4, linewidth = 2, color = 'r', linestyle = '--')
        plt.text(ax.get_xlim()[0]+2, capacity_max//4, f'capacity: {capacity_max//4}',
                 ha = 'left', va = 'center', color='r', backgroundcolor='w')
  if charge_limit:
        plt.axhline(y = battery_charge_limit, linewidth = 2, color = 'g', linestyle = '--', label = 'charge limit')
        plt.text(ax.get_xlim()[0]+2, battery_charge_limit, f'charge limit: {battery_charge_limit}',
                 ha='left', va='center', color='g', backgroundcolor='w')

  plt.show()


# exec

plot(
    multiple = False,
    capacity = False,
    charge_limit = False,
    output_limit = False
    )