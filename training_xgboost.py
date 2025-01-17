import xgboost as xgb
import data_processing as dpss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# reshape X
X_train = dpss.X_train.reshape(len(dpss.X_train), len(dpss.X_train[0]) * len(dpss.X_train[0][0]))
X_test = dpss.X_test.reshape(len(dpss.X_test), len(dpss.X_test[0]) * len(dpss.X_test[0][0]))

# build model
xgb_clf = xgb.XGBRegressor(base_score = 0.5,
                           booster = 'gbtree',
                           objective = 'reg:squarederror',
                           subsample = 1.0,
                           colsample_bytree = 1.0,
                           scale_pos_weight = 0.1,
                           )

kfold = KFold(n_splits = 10)

param_grids = {
    'n_estimators': [500, 1000],  # 500
    'learning_rate': [0.01, 0.05], # 0.05
    'max_depth': [3, 4, 5], #3
    'min_child_weight': [6, 8, 10], #10
    }

# grid search
xgb_model = GridSearchCV(xgb_clf,
                         param_grids,
                         n_jobs = -1,
                         scoring = 'neg_root_mean_squared_error',
                         cv = kfold,
                         verbose = 2)

# training
xgb_model.fit(X_train, dpss.y_train,
              verbose = 100, sample_weight = dpss.sample_weight)


print("Best hyperparameters: ", xgb_model.best_params_)
print("Best score: ", xgb_model.best_score_)

# prediction performance
y_pred = xgb_model.predict(X_test)
y_pred = dpss.y_scaler.inverse_transform(y_pred.reshape(-1,1))

dpss.pred_real_plot(y_pred, model = 'XGBoost')

# save model
xgb_model = xgb_model.best_estimator_
xgb_model.save_model('./models/xgb_model.ubj')



