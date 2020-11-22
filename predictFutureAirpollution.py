import pandas as pd
import pickle
import glob
import os
import matplotlib.pyplot as plt

base_path = "D:\Programming\Python\SustainabilityReport/"
pollutants = ['NO2', 'O3', 'PM10', 'PM25', 'SO2']
particle_limits = {'NO2': 40, 'O3': 000, 'PM10': 20, 'PM25': 10, 'SO2': 000}

region = "ES51"
model_name = "Random Forest"

historic_path = glob.glob(os.path.join(base_path, "historic", "{}_historic_*.csv".format(region)))[0]
forecast_path = glob.glob(os.path.join(base_path, "forecast", "{}_forecast_*_data.csv".format(region)))[0]

df_historic = pd.read_csv(historic_path)
df_historic = df_historic.drop(["NUTS_ID", "NUTS_NAME"], axis=1)
df_historic = df_historic.set_index("year")

df_forecast = pd.read_csv(forecast_path)
df_forecast = df_forecast.drop(["NUTS_ID", "NUTS_NAME"], axis=1)
df_forecast = df_forecast.set_index("year")

pollution_forcast_columns = pd.Series(pollutants)
pollution_forcast_columns = pollution_forcast_columns.append("upper_" + pd.Series(pollutants))
pollution_forcast_columns = pollution_forcast_columns.append("lower_" + pd.Series(pollutants))
df_pollution_forecast = pd.DataFrame(index=df_forecast.index, columns=pollution_forcast_columns)

columnOrder = pd.read_csv(os.path.join(base_path, "model", "ColumnsInOrder.csv"))

for pollutant in pollutants:
    model = pickle.load(open(os.path.join(base_path, "model", "model_{}_{}.pkl".format(pollutant, model_name)), 'rb'))
    for prefix in ["yhat_", "yhatupper_", "yhatlower_"]:
        selected_columns = [s for s in df_forecast.columns if prefix in s]
        df_temp = df_forecast[selected_columns]
        df_temp.columns = df_temp.columns.str.replace(prefix, '')
        df_temp = df_temp[columnOrder["column"].tolist()]
        y_pred = model.predict(df_temp)
        if prefix == "yhat_":
            df_pollution_forecast.loc[:, pollutant] = y_pred
        else:
            df_pollution_forecast.loc[:, "{}_{}".format(prefix[4:-1], pollutant)] = y_pred

df = df_historic.append(df_pollution_forecast)

#with confidence Intervall
##cmap = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
##for i, pollutant in enumerate(pollutants):
##    temp = df.filter(regex=pollutant)
##    plt.plot(temp.index, temp[pollutant], color=cmap[i])
##    plt.fill_between(temp.index, temp["upper_{}".format(pollutant)], temp["lower_{}".format(pollutant)], alpha=0.5, color=cmap[i])
##plt.legend(pollutants)
##plt.show()
#without confidence intervall
df[pollutants].plot()
plt.show()



