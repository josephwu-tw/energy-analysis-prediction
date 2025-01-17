import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import preprocessing as skpp
from sklearn.model_selection import train_test_split

''' Features ExplanaStion
PL  : production Line
CHP : chilled water pump
CH  : chiller
CWP : cooling water pump
CT  : cooling tower
AC  : air compressor
VM  : vacuum machine
EF  : exhaust fan
PCHP: production cooling water pump
'''

# dataframe setting 
raw_df = pd.read_csv('./dataset/powerconsumption.csv')
raw_df['Time'] = pd.to_datetime(raw_df['Time'], format = 'mixed')

# combine PL consumption
comb_df = raw_df.copy()
comb_df.insert(1, "PL", comb_df.iloc[:,1:9].sum(axis = 1))
comb_df.drop(comb_df.columns[2:10], axis = 1, inplace = True)

total_df = pd.concat([raw_df.iloc[:,:9], comb_df.iloc[:,1:]], axis = 1, join = "inner")

def get_df(df_type = 'comb'):

    if df_type == 'raw': return raw_df
    elif df_type == 'combine': return comb_df
    elif df_type == 'total': return total_df
    else:
        raise ValueError("Invalid value for 'df_type'. Expected 'raw', 'comeb' or 'total'.")

def get_group_train_test(X, Y, window_size: int, avg_add = True):
    data_len = X.shape[0]

    X_group = []
    Y_group = []

    for i in range(data_len-window_size):
        temp_group = X[i:i+window_size]

        if avg_add == True:
            avg_row = np.mean(temp_group, axis=0)
            temp_group = np.append(temp_group, [avg_row], axis = 0)

        X_group.append(temp_group)
        Y_group.append(Y[i+window_size])

    X_group = np.array(X_group)
    Y_group = np.array(Y_group)

    return X_group, Y_group


dataset = comb_df.copy()
dataset.insert(1, "Month", raw_df['Time'].apply(lambda x: x.month))
dataset.insert(2, "Weekday", raw_df['Time'].apply(lambda x: x.weekday()))
dataset.insert(3, "Hour", raw_df['Time'].apply(lambda x: x.hour))
dataset.drop(columns = ['Time', 'Others'], axis = 1, inplace = True)       

X_raw = dataset.to_numpy()
Y_raw = dataset.iloc[:,-1].to_numpy()

# add sample weight
sample_weight_raw = np.where(Y_raw >= 500 , 2, 1)

# normalize input
x_scaler = skpp.MinMaxScaler()
y_scaler = skpp.MinMaxScaler()
X = x_scaler.fit_transform(X_raw)
Y = y_scaler.fit_transform(Y_raw.reshape(-1,1))

# train test split
window_size = 4
X_group, Y_group = get_group_train_test(X, Y, window_size = window_size, avg_add = True)
X_train, X_test, y_train, y_test = train_test_split(X_group, Y_group,
                                                    test_size = 0.3,
                                                    shuffle = False)

y_test = y_scaler.inverse_transform(y_test.reshape(-1,1))


sample_weight = sample_weight_raw[window_size:len(y_train)+window_size]

def pred_real_plot(y_pred, model, y_test = y_test, dataframe = raw_df, multiple = False):
    palette = list(sns.palettes.mpl_palette('Dark2'))
    data_len = len(dataframe)
    x = dataframe['Time'][(data_len - len(y_test)):]

    fig, ax = plt.subplots(figsize = (20,10))
    test = y_test
    ax.plot(x, test)

    if multiple == True:
        for pred in y_pred:
            ax.plot(x, pred, ls = (0, (2, 2)))
        
        ax.set(xlabel = 'Time', ylabel = 'kWh',
                title = f'Prediction vs Real')
        ax.legend(['Real'] + [name for name in model])

    else:
        pred = y_pred

        ax.plot(x, pred)

        ax.set(xlabel = 'Time', ylabel = 'kWh',
                title = f'Prediction vs Real - {model}')
        ax.legend(['Real', 'Prediction'])

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.show()

