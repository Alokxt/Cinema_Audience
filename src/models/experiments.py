from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np 
import matplotlib.pyplot as plt

from src.models.train_test import get_train_test_data
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*"
)

import pandas as pd 

Lr = LinearRegression()
Rd = Ridge()
las = Lasso()
dc = DecisionTreeRegressor()
Rnd = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
Grd = GradientBoostingRegressor(n_estimators=200, random_state=42)
Extr = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
ada = AdaBoostRegressor(n_estimators=200, random_state=42)
xgb = XGBRegressor(n_estimators=200, random_state=42,learning_rate=0.01,max_depth=8)
lgbm = LGBMRegressor(n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,)

models = {
    "Linear_Regression":Lr,
    "Ridge":Rd,
    "Lasso":las,
    "DecisionTree":dc,
    "RandomForest":Rnd,
    "Xgboost":xgb,
    "LightGBM":lgbm
}

def predict_test_data(X_test,m,pr,l1,l7):
    preds = np.zeros(X_test.shape[0])
    
    for i in range(X_test.shape[0]):
      
      X_test[i][12] = l1
      X_test[i][13] = l7
      X_test[i][1]  =pr
      pred = int(m.predict(X_test[[i]])[0])
      preds[i] = pred 
      
      l1 = pred 
      if i>7:
          l7 = preds[i-6]
      pr = preds[:i+1].mean()
    return preds 
    

def model_evals(X_tr,Y_tr,X_ts,Y_ts,models):
    all_r2_scores_train = {}
    all_rmse_erroes_train = {}
    all_r2_scores_test = {}
    all_rmse_erroes_test = {}
    for i in list(models.keys()):
        print(f'Model number {i} performace')
        models[i].fit(X_tr,Y_tr)
        pred_train = models[i].predict(X_tr)
        mse_train = mean_squared_error(Y_tr,pred_train)
        r2_train = r2_score(Y_tr,pred_train)
        pr = X_tr[-1][1]
        l1 = X_tr[-1][12]
        l7 = X_tr[-1][13]
        pred_test  = predict_test_data(X_ts,models[i],pr,l1,l7)
        mse_test = mean_squared_error(Y_ts,pred_test)
        r2_test = r2_score(Y_ts,pred_test)
        all_rmse_erroes_train[i] = mse_train**0.5 
        all_r2_scores_train[i] = r2_train
        all_rmse_erroes_test[i] = mse_test**0.5 
        all_r2_scores_test[i] = r2_test
        print(f'training errors ,rmse {mse_train**0.5} r2_score {r2_train}')
        print(f'testing errors rmse  {mse_test**0.5} r2_score {r2_test}')
    return all_rmse_erroes_train,all_r2_scores_train,all_rmse_erroes_test,all_r2_scores_test








def plot_results(results_df):
    x = np.arange(len(results_df["Model"]))
    width = 0.35

    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, results_df["RMSE_Train"], width, label="Train RMSE")
    plt.bar(x + width/2, results_df["RMSE_Test"], width, label="Test RMSE")

    plt.xticks(x, results_df["Model"], rotation=30)
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison Across Models")
    plt.legend()

    plt.tight_layout()
    plt.savefig("C:\\Users\\Nimisha Manawat\\OneDrive\\Desktop\\CinemaAudience2\\src\\plots\\rmse_comp",
            dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, results_df["R2_Train"], width, label="Train R²")
    plt.bar(x + width/2, results_df["R2_Test"], width, label="Test R²")

    plt.xticks(x, results_df["Model"], rotation=30)
    plt.ylabel("R² Score")
    plt.title("R² Score Comparison Across Models")
    plt.legend()

    plt.tight_layout()
    plt.savefig("C:\\Users\\Nimisha Manawat\\OneDrive\\Desktop\\CinemaAudience2\\src\\plots\\r2_comp",
            dpi=300, bbox_inches="tight")
    plt.show()


X_tr,Y_tr,X_ts,Y_ts = get_train_test_data()



rmse_tr, r2_tr, rmse_ts, r2_ts = model_evals(
    X_tr, Y_tr, X_ts, Y_ts, models
)

results_df = pd.DataFrame({
    "Model": rmse_tr.keys(),
    "RMSE_Train": rmse_tr.values(),
    "RMSE_Test": rmse_ts.values(),
    "R2_Train": r2_tr.values(),
    "R2_Test": r2_ts.values()
})

plot_results(results_df)


"""
***************************************************************************************************
Model number RandomForest performace
training errors ,rmse 8.583640504060636 r2_score 0.9336194439446516
testing errors rmse  26.115126583779027 r2_score 0.2772720611156366
***************************************************************************************************
Model number ExtraTrees performace
training errors ,rmse 0.0 r2_score 1.0
testing errors rmse  24.434876908437325 r2_score 0.3672809878804133

***************************************************************************************************
Model number AdaBoost performace
training errors ,rmse 30.128516681688918 r2_score 0.18218824939622213
testing errors rmse  30.9629217415775 r2_score -0.015954933153215833

***************************************************************************************************
Model number GradientBoosting performace
training errors ,rmse 23.62155527740497 r2_score 0.49729312713189844
testing errors rmse  21.977564955294007 r2_score 0.48814173781830783

***************************************************************************************************
Model number XGBoost performace
training errors ,rmse 22.357750014625115 r2_score 0.5496460199356079
testing errors rmse  32.789581358162245 r2_score -0.13936353780756305

***************************************************************************************************
Model number LightGBM performace
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.009873 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 1259
[LightGBM] [Info] Number of data points in the train set: 171236, number of used features: 27
[LightGBM] [Info] Start training from score 42.216707
training errors ,rmse 22.656706573724847 r2_score 0.5375216514614016
testing errors rmse  21.81609335916343 r2_score 0.49563546601987096
"""



def get_final_model():
    lgbm = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    lgbm.fit(X_tr,Y_tr)

    return lgbm