import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from data_processing import get_df

total_df = get_df(df_type = 'total')

# battery output max
battery_output_max = 45

# get description
description = total_df.describe().iloc[1:7,1:21].T.replace(0,0.001).apply(lambda x: round(x,3))

# description.T.to_csv('features_description.csv')

# get support hours
support_hours = description.apply(lambda x: round(battery_output_max*2/x,1))

# plotting
features = ['PL', 'D_Building', 'CWP', 'CT', 'AC', 'VM', 'EF']

ax = support_hours.loc[features[::-1]]['75%'].plot.barh(title = 'Supporting Hours for 75% power consumption',
                                      ylabel = 'Features',
                                      xlabel = 'hrs',
                                      xlim = (0,10),
                                      figsize = (10,5),
                                      rot = 45)
ax.bar_label(ax.containers[0], padding = 2)
plt.axvline(x = 2, linewidth = 2, color = 'r', linestyle = '--')
plt.show()
