import xgboost as xgb
import keras
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
import data_processing as dpss
import pandas as pd
from sklearn import preprocessing as skpp

# function
def model_eval(y_pred, y_test):
    # rmse
    rmse = root_mean_squared_error(y_test, y_pred)

    # average high peak (>=500) difference
    cnt = 0
    summ = 0

    for pred, real in zip(y_pred.reshape(-1), y_test.reshape(-1)):
        if real >= 500:
          summ += (pred - real)
          cnt += 1

    highpeak_diff = summ/cnt

    # difference between highest real with related prediction
    max_idx = y_test.reshape(-1).argmax()
    max_diff = y_pred.reshape(-1)[max_idx] - y_test.reshape(-1)[max_idx]

    return round(rmse, 3), round(highpeak_diff, 3), round(max_diff, 3)

# get data
X_test_lstm = dpss.X_test 
X_test = dpss.X_test.reshape(len(dpss.X_test), len(dpss.X_test[0]) * len(dpss.X_test[0][0]))

# load models
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("./models/xgb_model.ubj")

lgbm_model = lgb.Booster(model_file = "./models/lgbm_model.txt")

keras.backend.clear_session()
lstm_model = keras.models.load_model("./models/lstm_model.keras")

# prediction original data
xgb_pred = xgb_model.predict(X_test)
xgb_pred = dpss.y_scaler.inverse_transform(xgb_pred.reshape(-1,1))
xgb_rmse, xgb_highpeak_diff, xgb_max_diff = model_eval(xgb_pred, dpss.y_test)

lgbm_pred = lgbm_model.predict(X_test)
lgbm_pred = dpss.y_scaler.inverse_transform(lgbm_pred.reshape(-1,1))
lgbm_rmse, lgbm_highpeak_diff, lgbm_max_diff = model_eval(lgbm_pred, dpss.y_test)

lstm_pred = lstm_model.predict(X_test_lstm)
lstm_pred = dpss.y_scaler.inverse_transform(lstm_pred.reshape(-1,1))
lstm_rmse, lstm_highpeak_diff, lstm_max_diff = model_eval(lstm_pred, dpss.y_test)

# output table
performance_table = pd.DataFrame({
   'Model': ['XGBoost', 'LightGBM', 'LSTM'],
   'RMSE': [xgb_rmse, lgbm_rmse, lstm_rmse],
   'Highpeak Diff': [xgb_highpeak_diff, lgbm_highpeak_diff, lstm_highpeak_diff],
   'Diff at Max': [xgb_max_diff, lgbm_max_diff, lstm_max_diff]
})

dpss.pred_real_plot(y_pred = [xgb_pred, lgbm_pred, lstm_pred],
                    model = ['XGBoost', 'LightGBM', 'LSTM'],
                    multiple = True)

performance_table.to_csv('model_evaluation_original.csv', index = False)

# import new data

new_df = pd.read_csv('./dataset/powerconsumption_new.csv')
new_df['Time'] = pd.to_datetime(new_df['Time'])
new_df.insert(1, "PL", new_df.iloc[:,1:9].sum(axis = 1))
new_df.drop(new_df.columns[2:10], axis = 1, inplace = True)
new_df.insert(1, "Month", new_df['Time'].apply(lambda x: x.month))
new_df.insert(2, "Weekday", new_df['Time'].apply(lambda x: x.weekday()))
new_df.insert(3, "Hour", new_df['Time'].apply(lambda x: x.hour))

dataset = new_df.copy()
dataset.drop(columns = ['Time', 'Others'], axis = 1, inplace = True)

X = dataset.to_numpy()
Y = dataset.iloc[:,-1].to_numpy()

x_scaler = skpp.MinMaxScaler()
y_scaler = skpp.MinMaxScaler()
X = x_scaler.fit_transform(X)
Y = y_scaler.fit_transform(Y.reshape(-1,1))

X_new_lstm, y_new = dpss.get_group_train_test(X = X, Y = Y, window_size = dpss.window_size)

y_new = y_scaler.inverse_transform(y_new.reshape(-1,1))

X_new = X_new_lstm.reshape(len(X_new_lstm), len(X_new_lstm[0]) * len(X_new_lstm[0][0]))

# new data prediction
xgb_pred_new = xgb_model.predict(X_new)
xgb_pred_new = y_scaler.inverse_transform(xgb_pred_new.reshape(-1,1))
xgb_rmse_new, xgb_highpeak_diff_new, xgb_max_diff_new = model_eval(xgb_pred_new, y_new)

lgbm_pred_new = lgbm_model.predict(X_new)
lgbm_pred_new = y_scaler.inverse_transform(lgbm_pred_new.reshape(-1,1))
lgbm_rmse_new, lgbm_highpeak_diff_new, lgbm_max_diff_new = model_eval(lgbm_pred_new, y_new)

lstm_pred_new = lstm_model.predict(X_new_lstm)
lstm_pred_new = y_scaler.inverse_transform(lstm_pred_new.reshape(-1,1))
lstm_rmse_new, lstm_highpeak_diff_new, lstm_max_diff_new = model_eval(lstm_pred_new, y_new)

# output table
performance_table_new = pd.DataFrame({
   'Model': ['XGBoost', 'LightGBM', 'LSTM'],
   'RMSE': [xgb_rmse_new, lgbm_rmse_new, lstm_rmse_new],
   'Highpeak Diff': [xgb_highpeak_diff_new, lgbm_highpeak_diff_new, lstm_highpeak_diff_new],
   'Diff at Max': [xgb_max_diff_new, lgbm_max_diff_new, lstm_max_diff_new]
})

dpss.pred_real_plot(y_pred = [xgb_pred_new, lgbm_pred_new, lstm_pred_new],
                    y_test = y_new,
                    dataframe = new_df,
                    model = ['XGBoost', 'LightGBM', 'LSTM'],
                    multiple = True)

performance_table_new.to_csv('model_evaluation_new.csv', index = False)