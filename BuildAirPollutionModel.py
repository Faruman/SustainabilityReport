import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,  AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

base_path = "D:\Programming\Python\SustainabilityReport/"
pollutants = ['NO2', 'O3', 'PM10', 'PM25', 'SO2']

linreg = LinearRegression()
polyreg = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
elasticnet = ElasticNet()
lassoreg = Lasso(positive= True)
dectree = DecisionTreeRegressor()

randfst = RandomForestRegressor()
adaboost =  AdaBoostRegressor()

models = {"Linear Regression": linreg, "Polynomial Regression": polyreg, "ElasticNet Regression": elasticnet, "Lasso Regression with positive Coefficients": lassoreg, "Decision Tree": dectree, "Random Forest": randfst, "AdaBoost": adaboost}

df = pd.read_csv(os.path.join(base_path + "data\data.csv"))

df = df.set_index(['NUTS_ID', 'year'])
df_NUTS_name = df["NUTS_NAME"]
df_y = df[pollutants]
df_X = df[list(set(df.columns) - set(["NUTS_NAME"]) - set(pollutants))]

results = pd.DataFrame(index= range(0, len(pollutants) * len(models.keys())), columns=["model", "pollutant", "mse", "r2"])
feature_importance_decTree = pd.DataFrame(index= pollutants, columns= df_X.columns)
feature_importance_linReg = pd.DataFrame(index= pollutants, columns= ["intercept"] + list(df_X.columns))
feature_importance_lassoReg = pd.DataFrame(index= pollutants, columns= ["intercept"] + list(df_X.columns))

i = 0

for pollutant in pollutants:
    j = 1
    y = df_y[pollutant]
    y = y.dropna()
    X = df_X.loc[y.index, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    y_bcn = y.loc[["ES51"]]
    X_bcn = X.loc[["ES51"]]
    if i == 0:
        pd.Series(X.columns, name="column").to_csv(os.path.join(base_path, "model", "ColumnsInOrder.csv"), index= False)
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15,15))
    ax[0][0].plot(y_bcn.index.get_level_values(1), y_bcn.values, 'g')
    ax[0][0].set_title("Reality", fontsize=12)
    print("{} Observations: {}".format(pollutant, y.shape[0]))
    for model_name in models.keys():
        print("     Creating {} for {}".format(model_name, pollutant))
        model = models[model_name]
        model.fit(X_train.reset_index(drop= True), y_train)
        y_pred = model.predict(X_test.reset_index(drop= True))
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print('     Mean squared error: %.2f' % mse)
        print('     Coefficient of determination: %.2f' % r2)
        results.iloc[i, :] = [model_name, pollutant, mse, r2]
        i += 1
        pkl_filename = os.path.join(base_path, "model", "model_{}_{}.pkl".format(pollutant, model_name))
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        if model_name == "Decision Tree":
            feature_importance_decTree.loc[pollutant, :] = model.feature_importances_
        if model_name == "Linear Regression":
            feature_importance_linReg.loc[pollutant, "intercept"] = model.intercept_
            feature_importance_linReg.loc[pollutant, df_X.columns] = model.coef_
        if model_name == "Lasso Regression with positive Coefficients":
            feature_importance_lassoReg.loc[pollutant, "intercept"] = model.intercept_
            feature_importance_lassoReg.loc[pollutant, df_X.columns] = model.coef_
        y_pred_bcn = model.predict(X_bcn)
        col_idx = j % 2
        row_idx = int(j / 2)
        ax[row_idx][col_idx].plot(y_bcn.index.get_level_values(1), y_pred_bcn)
        ax[row_idx][col_idx].set_title("{} (R2: {:.2f})".format(model_name, r2), fontsize=12)
        j += 1
    fig.suptitle("Different Models for {}".format(pollutant), fontsize=20)
    plt.savefig(os.path.join(base_path, "model", "CompareModels_{}.png".format(pollutant)))
    plt.show()

results.to_excel(os.path.join(base_path, "model", "results.xlsx"), index= False)
feature_importance_decTree = feature_importance_decTree.reindex(sorted(feature_importance_decTree.columns), axis=1)
feature_importance_decTree.to_excel(os.path.join(base_path, "model", "FeatureImportanceDecTree.xlsx"))
feature_importance_linReg = feature_importance_linReg.reindex(sorted(feature_importance_linReg.columns), axis=1)
feature_importance_linReg.to_excel(os.path.join(base_path, "model", "FeatureImportanceLinReg.xlsx"))
feature_importance_lassoReg = feature_importance_lassoReg.reindex(sorted(feature_importance_lassoReg.columns), axis=1)
feature_importance_lassoReg.to_excel(os.path.join(base_path, "model", "FeatureImportanceLassoReg.xlsx"))

