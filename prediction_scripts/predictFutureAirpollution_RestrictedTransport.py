import pandas as pd
import numpy as np
import pickle
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

base_path = "D:\Programming\Python\SustainabilityReport/"
pollutants = ['NO2', 'O3', 'PM10', 'PM25', 'SO2']
particle_limits = {'NO2': 40, 'O3': 100, 'PM10': 20, 'PM25': 10, 'SO2': 20}

region = "ES51"
model_name = "Decision Tree"
policy_dict = {"vreg_CAR": -0.5}

historic_path = glob.glob(os.path.join(base_path, "historic", "{}_historic_*_V2.csv".format(region)))[0]
forecast_path = glob.glob(os.path.join(base_path, "forecast", "{}_forecast_*_data_V2.csv".format(region)))[0]

df_historic = pd.read_csv(historic_path)
df_historic = df_historic.drop(["NUTS_ID", "NUTS_NAME"], axis=1)
df_historic = df_historic.set_index("year")

df_forecast = pd.read_csv(forecast_path)
df_forecast = df_forecast.drop(["NUTS_ID", "NUTS_NAME"], axis=1)
df_forecast = df_forecast.set_index("year")

for key in policy_dict.keys():
    filter_columns = [s for s in df_forecast.columns if key in s]
    base = df_forecast.loc[df_forecast.index.min(), filter_columns]
    forecast = base + (base * policy_dict[key])
    slope = (forecast - base) / (df_forecast.index.max() - df_forecast.index.min())
    for year in list(set(df_forecast.index) - set([df_forecast.index.min()])):
        df_forecast.loc[year, filter_columns] = base + slope * (year - df_forecast.index.min())

pollution_forcast_columns = pd.Series(pollutants)
pollution_forcast_columns = pollution_forcast_columns.append("upper_" + pd.Series(pollutants))
pollution_forcast_columns = pollution_forcast_columns.append("lower_" + pd.Series(pollutants))
df_pollution_forecast = pd.DataFrame(index=df_forecast.index, columns=pollution_forcast_columns)

columnOrder = pd.read_csv(os.path.join(base_path, "model", "ColumnsInOrder_V2.csv"))

for pollutant in pollutants:
    model = pickle.load(open(os.path.join(base_path, "model", "model_{}_{}_V2.pkl".format(pollutant, model_name)), 'rb'))
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
plt.title("Devlopment air pollution 1990 to 2030 - restrict traffic")
plt.savefig(os.path.join(base_path, "plots", "airPollution19902030_restTraffic_V2.png"))
plt.show()

# create graph which shows level above the threshold in red
cmap = ["C0", "C1", "C2", "C4", "C6", "C7", "C8", "C9"]
c_above_th = "C3"
for i, pollutant in enumerate(pollutants):
    temp = df.filter(regex=pollutant)
    x = np.array(temp.index)
    y = temp.loc[:, pollutant].values

    thres_passes = np.where((y[~np.isnan(y)]>particle_limits[pollutant])[:-1] != (y[~np.isnan(y)]>particle_limits[pollutant])[1:])[0]
    for thres_pass in thres_passes:
        slope = 1/ (y[~np.isnan(y)][thres_pass+1] - y[~np.isnan(y)][thres_pass])
        x_new = slope * (particle_limits[pollutant] - y[~np.isnan(y)][thres_pass]) + x[~np.isnan(y)][thres_pass]
        x = np.append(x, x_new)
        y = np.append(y, particle_limits[pollutant]+ 1/(10**5) * slope)
    x_inds = x.argsort()
    x = x[x_inds]
    y = y[x_inds]
    colormap = ListedColormap([cmap[i], c_above_th])
    highest_value = max(np.nanmax(y), particle_limits[pollutant])
    norm = BoundaryNorm([0, particle_limits[pollutant], highest_value], colormap.N)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=colormap, norm=norm)
    lc.set_array(y)
    lc.set_linewidth(2)

    plt.gca().add_collection(lc)
    plt.xlim(x.min(), x.max())
    plt.ylim(0, highest_value*1.1)
    plt.hlines(particle_limits[pollutant], x[0], x[-1], colors='k', linestyles='solid', label='WHO threshold')
    plt.title("Development of {} (with restricted traffic)".format(pollutant))
    plt.xlabel('year')
    plt.ylabel('µg/m³')
    handles, labels = plt.gca().get_legend_handles_labels()
    patch = mpatches.Patch(color=cmap[i], label=pollutant)
    handles.insert(0, patch)
    plt.legend(handles=handles)
    plt.savefig(os.path.join(base_path, "plots", "Development_{}_{:.0f}_{:.0f}_restTraffic_V2.png".format(pollutant, x.min(), x.max())))
    plt.show()

