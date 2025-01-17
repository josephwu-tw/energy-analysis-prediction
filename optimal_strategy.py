import numpy as np
import pandas as pd
import datetime as dt
from data_processing import get_df
import price_table_time as ptt

df = get_df('raw')
low_bound = 1800
high_bound = 2400
step = 50

curr_capacity = 2400
curr_stype = 2

battery = 400
charge_rate = 0.5
lose_rate = 0.9
renew_price = 5.7
renew_factor = 1250

# Functions
def is_summer(time: pd.Timestamp) -> bool:
    return (ptt.summer_end >= time >= ptt.summer_start)

def is_offday(time: pd.Timestamp) -> bool:
    return (time.weekday() == 6) or (time.strftime('%Y-%m-%d') in ptt.off_peak_days)

def find_optimal(base, low_bound = low_bound, high_bound = high_bound, step = step, type_change = True):
    optimal = base
    save_cost = 0

    type_range = [2, 3] if type_change == True else [base.section_type]
    for stype in type_range:

        for cap in range(low_bound, high_bound + step, step):
            new = PriceSet(section_type = stype, capacity = cap)

            if new.cost < optimal.cost:
                save_cost = round(base.cost - new.cost, 3)
                optimal = new

    return (optimal, save_cost)


def battery_benefit(dataframe = df, capacity = curr_capacity, battery = battery, section_type = curr_stype, pricediff_multiplier = 1.0):
    if section_type not in [2, 3]:
        raise ValueError("Invalid value for 'section_type'. Expected 2 or 3.")

    table = ptt.price_section_2 if section_type == 2 else ptt.price_section_3

    dates = dataframe.groupby(pd.Grouper(key = 'Time', freq = '1D')).groups.keys()
    #dates = pd.date_range('2024-01-01', '2024-12-31', freq ='D')

    summer_weekday = sum(1 for date in dates if (is_summer(date) and not is_offday(date) and date.weekday() < 5))
    summer_sat = sum(1 for date in dates if (is_summer(date) and not is_offday(date) and date.weekday() == 5))
    summer_offday = sum(1 for date in dates if (is_summer(date) and is_offday(date)))
    normal_weekday = sum(1 for date in dates if (not is_summer(date) and not is_offday(date) and date.weekday() < 5))
    normal_sat = sum(1 for date in dates if (not is_summer(date) and not is_offday(date) and date.weekday() == 5))

    # maintain cost
    maintain_cost = 630000 * (len(dates)/366)

    # price benefit
    summer_weekday_diff = (table['summer_highpeak'] if section_type == 3 else table['summer_peak']) - table['summer_offpeak']
    summer_sat_diff = table['summer_sat'] - table['summer_offpeak']
    normal_weekday_diff = table['normal_peak'] - table['normal_offpeak']
    normal_sat_diff = table['normal_sat'] - table['normal_offpeak']

    summer_save = ((summer_weekday*summer_weekday_diff) + (summer_sat*summer_sat_diff)) * battery * pricediff_multiplier
    normal_save = ((normal_weekday*normal_weekday_diff) + (normal_sat*normal_sat_diff)) * battery * 2 * pricediff_multiplier
    price_save = summer_save + normal_save

    # renew benefit
    renew_origin = (capacity*0.1) * renew_factor * (len(dates)/366)
    battery_buy_save = (battery*charge_rate) * renew_factor * (len(dates)/366)
    renew_cost_save = battery_buy_save * renew_price * pricediff_multiplier
    renew_daily_diff = battery_buy_save // len(dates) + 1
    renew_credit_loss = (summer_weekday*table['summer_peak'] + \
                         summer_sat*table['summer_sat'] + \
                         summer_offday*table['summer_offpeak'] + \
                         (len(dates)-(summer_weekday+summer_sat +summer_offday))*table['normal_offpeak']) * renew_daily_diff * pricediff_multiplier

    # calculate total benefit
    total_benefit = price_save + renew_cost_save - maintain_cost - renew_credit_loss

    # output
    output = [pricediff_multiplier, section_type, battery,
              normal_save, summer_save, price_save,
              renew_cost_save, renew_credit_loss,
              maintain_cost, total_benefit]

    return [f'{pricediff_multiplier} times'] + [f'{res}' for res in output[1:3]] + [f'{res:,.2f}' for res in output[3:]]

# PriceSet Class
class PriceSet:

      def __init__(self, section_type: int, capacity: int , multiplier = 1):
          if section_type not in [2, 3]:
              raise ValueError("Invalid value for 'section_type'. Expected 2 or 3.")

          self.section_type = section_type
          self.price_table = ptt.price_section_2 if self.section_type == 2 else ptt.price_section_3
          self.capacity = capacity
          self.dates = pd.date_range(df['Time'].iloc[0], df['Time'].iloc[-1], freq = 'D')
          self.exceed_month = 0
          self.exceed_time = 0
          self.extra_fee = 0
          self.basic_fee = 0
          self.float_fee = 0
          self.renew_cost, self.renew_credit = self.get_renew_credit()

          self.cost = self.get_total_cost()

      def stat(self, detail = False):
          print(f"{'-'*60}")
          print(f"# Section type : {self.section_type}\n" +
                f"# Capacity     : {self.capacity:,}\n" +
                f"# Cost         : {self.cost:,.3f}")

          if detail:
              print('--- Details ---')
              print(f"+ Basic Fee    : {self.basic_fee:,.3f}\n" +
                    f"+ Floating Fee : {self.float_fee:,.3f}\n" +
                    f"+ Renew Cost   : {self.renew_cost:,.3f}\n" +
                    f"- Renew Credit : {self.renew_credit:,.3f}\n" +
                    f"+ Extra  Fee   : {self.extra_fee:,.3f}\n" +
                    f"* Exceed Charge: {self.exceed_month} months\n" +
                    f"* Exceed Time  : {self.exceed_time} times")
          print(f"{'-'*60}")

      def get_renew_credit(self) -> int:

          global renew_factor
          global renew_price

          summer_weekday = sum(1 for date in self.dates if (is_summer(date) and not is_offday(date) and date.weekday() < 5))
          summer_sat = sum(1 for date in self.dates if (is_summer(date) and not is_offday(date) and date.weekday() == 5))
          summer_offday = sum(1 for date in self.dates if (is_summer(date) and is_offday(date)))
          normal_weekday = sum(1 for date in self.dates if (not is_summer(date) and not is_offday(date) and date.weekday() < 5))
          normal_sat = sum(1 for date in self.dates if (not is_summer(date) and not is_offday(date) and date.weekday() == 5))
          renew_buy = (self.capacity*0.1) * renew_factor * (len(self.dates)/365)
          renew_cost = renew_buy * renew_price
          renew_daily = renew_buy // len(self.dates) + 1
          renew_credit = (summer_weekday*self.price_table['summer_peak'] + \
                          summer_sat*self.price_table['summer_sat'] + \
                          summer_offday*self.price_table['summer_offpeak'] + \
                           (len(self.dates)-(summer_weekday+summer_sat+summer_offday))*self.price_table['normal_offpeak']) * renew_daily

          return renew_cost, renew_credit

      def get_total_cost(self) -> int:

          total_cost = 0
          start_month = df.iloc[0]['Time'].month
          end_month = df.iloc[-1]['Time'].month
          for month in range(start_month, end_month + 1):
              month_df = df[df['Time'].dt.month == month]
              total_cost += self.get_monthly_cost(month_df)

          return total_cost - self.renew_cost + self.renew_credit

      def get_monthly_cost(self, month_df) -> int:

          # exceed capacity fee
          max_usage = month_df.max()
          extra_fee_price = ptt.basic_price['summer'] if is_summer(max_usage['Time']) else ptt.basic_price['normal']
          max_usage = max_usage['Total']
          if max_usage * 4 > self.capacity * 1.1:
              extra_fee = extra_fee_price * (max_usage * 12 - self.capacity * 3.1)
          elif self.capacity * 1.1 >= max_usage * 4 > self.capacity:
              extra_fee = extra_fee_price * (max_usage * 4 - self.capacity)
          else:
              extra_fee = 0

          extra_fee = round(extra_fee, 3)
          self.extra_fee += extra_fee
          self.exceed_month += 1 if extra_fee > 0 else 0

          # basic fee
          days = pd.date_range(month_df['Time'].iloc[0].date(), month_df['Time'].iloc[-1].date(), freq ='D')
          summer_days = sum(1 for day in days if is_summer(day))
          normal_days = sum(1 for day in days if not is_summer(day))
          total_days = summer_days + normal_days
          basic_fee = round((summer_days * ptt.basic_price['summer'] +
                            normal_days * ptt.basic_price['normal']) / total_days, 3) * self.capacity
          self.basic_fee += basic_fee

          # floating fee
          float_fee = 0

          for idx, row in month_df.iterrows():
              float_fee += self.get_float_cost(row)
              if row['Total'] * 4 > self.capacity:
                  self.exceed_time += 1

          float_fee = round(float_fee, 3)
          self.float_fee += float_fee

          return round(extra_fee + basic_fee + float_fee, 3)

      def get_float_cost(self, data) -> int:
          price = self.get_float_price(data['Time'])
          usage = data['Total']

          return price * usage

      def get_float_price(self, time) -> float:

          hour = time.hour
          weekday = time.weekday()

          global is_summer
          global is_offday

          # define pricing situation
          summer = is_summer(time)
          offday = is_offday(time)
          peak_hour = (23 >= hour >= 9) if summer else (10 >= hour >= 6) or (23 >= hour >= 14)
          is_peak = peak_hour and (not offday)

          high_peak = (21 >= hour >= 16)

          # price picking logic
          if summer and is_peak:
              if self.section_type == 3 and high_peak:
                  return self.price_table['summer_highpeak']
              else:
                  return self.price_table['summer_sat'] if weekday == 5 else self.price_table['summer_peak']
          elif summer and not is_peak:
              return self.price_table['summer_offpeak']
          elif not summer and is_peak:
              return self.price_table['normal_sat'] if weekday == 5 else self.price_table['normal_peak']
          else:
              return self.price_table['normal_offpeak']


current = PriceSet(section_type = curr_stype, capacity = curr_capacity)

print('###  Current Strategy  ###')
current.stat(detail = True)

# without sectiontype change
optimal_capacity, save_cost_capacity = find_optimal(current, type_change = False)

print('')
print('###  Optimal Strategy - same section type  ###')
optimal_capacity.stat(detail = True)
print(f'> Save Cost    : {save_cost_capacity:,.3f}')

# with sectiontype change
optimal_strategy, save_cost_strategy = find_optimal(current, type_change = True)

print('')
print('###  Optimal Strategy - change section type  ###')
optimal_strategy.stat(detail = True)
print(f'> Save Cost    : {save_cost_strategy:,.3f}')


# Battery Benefit

best_capacity = optimal_strategy.capacity

benefit_table = pd.DataFrame(columns = ['price_differ', 'section_type', 'battery',
                                        'normal_save', 'summer_save', 'price_save',
                                        'renew_cost_save', 'renew_credit_loss',
                                        'maintain_cost', 'total_benefit'])

for stype in [2, 3]:
    for diff in [1.0, 1.2, 1.6, 2.0]:
        benefit_table.loc[len(benefit_table)] = battery_benefit(section_type = stype, 
                                                                pricediff_multiplier = diff, 
                                                                capacity = best_capacity,
                                                                )
benefit_table.to_csv('battery_benefit_table.csv', index = False)