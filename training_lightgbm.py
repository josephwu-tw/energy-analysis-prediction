import lightgbm as lgb
import data_processing as dpss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# reshape X
X_train = dpss.X_train.reshape(len(dpss.X_train), len(dpss.X_train[0]) * len(dpss.X_train[0][0]))
X_test = dpss.X_test.reshape(len(dpss.X_test), len(dpss.X_test[0]) * len(dpss.X_test[0][0]))

# build model
lgbm_clf = lgb.LGBMRegressor(boosting_type = 'gbdt',
                             num_leaves = 5)

kfold = KFold(n_splits = 10)

param_grids = {
    'n_estimators': [500, 1000],  # 1000
    'learning_rate': [0.01, 0.05], # 0.05
    'max_depth': [3, 4, 5], # 5
    'min_child_weight': [6, 8, 10] # 8
    }

# grid search
lgbm_model = GridSearchCV(lgbm_clf,
                          param_grids,
                          n_jobs = -1,
                          scoring = 'neg_root_mean_squared_error',
                          cv = kfold
                          )

# training
lgbm_model.fit(X_train, dpss.y_train,
               sample_weight = dpss.sample_weight)

print("Best hyperparameters: ", lgbm_model.best_params_)
print("Best score: ", lgbm_model.best_score_)

# prediction performance
y_pred = lgbm_model.predict(X_test)
y_pred = dpss.y_scaler.inverse_transform(y_pred.reshape(-1,1))

dpss.pred_real_plot(y_pred, model = 'LightGBM')

# save model
lgbm_model = lgbm_model.best_estimator_
lgbm_model.booster_.save_model('./models/lgbm_model.txt')