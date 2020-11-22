import os
import pandas as pd
from datetime import datetime, timedelta
from fbprophet import Prophet

base_path = "D:\Programming\Python\SustainabilityReport/"
pollutants = ['NO2', 'O3', 'PM10', 'PM25', 'SO2']
predict_years = range(2020, 2031)
filter_region = "ES51"

df = pd.read_csv(os.path.join(base_path + "data\data.csv"))

df = df.loc[df["NUTS_ID"] == filter_region]
start_year = df["year"].min()
end_year = df["year"].max()
df[["NUTS_ID", "NUTS_NAME", "year"] + pollutants].to_csv(os.path.join(base_path, "historic", "{}_historic_{}_{}.csv".format(filter_region, start_year, end_year)), index= False)

df = df[list(set(df.columns) - set(pollutants))]

columns = pd.Series(list(set(df.columns) - set(['NUTS_ID', 'NUTS_NAME', 'year'])))
predict_columns = "yhat_" + columns
predict_columns = predict_columns.append("yhatlower_" + columns)
predict_columns = predict_columns.append("yhatupper_" + columns)
df_predicted = pd.DataFrame(columns= pd.Series(["NUTS_ID", "NUTS_NAME", "year"]).append(predict_columns))
df_predicted["year"] = predict_years
df_predicted["NUTS_ID"] = filter_region
df_predicted["NUTS_NAME"] = df["NUTS_NAME"].iloc[0]
df_predicted["ds"] = pd.to_datetime(pd.Series(predict_years) + 1, format='%Y') - timedelta(days=1)

df["ds"] = pd.to_datetime(df["year"] + 1, format='%Y') - timedelta(days=1)

for column in list(columns):
    df_temp = df[["ds", column]]
    df_temp.columns = ["ds", "y"]
    model = Prophet()
    model.fit(df_temp)
    future = pd.DataFrame(index= df_predicted.index, columns=["ds"])
    future["ds"] = df_predicted["ds"]
    forecast = model.predict(future)
    df_predicted.loc[:, ["yhat_{}".format(column), "yhatlower_{}".format(column), "yhatupper_{}".format(column)]] = forecast[["yhat", "yhat_lower", "yhat_upper"]].values

df_predicted = df_predicted.drop("ds", axis= 1)
df_predicted.to_csv(os.path.join(base_path, "forecast", "{}_forecast_{}_{}_data.csv".format(filter_region, predict_years[0], predict_years[-1])), index=False)