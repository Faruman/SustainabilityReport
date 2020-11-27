from FutureAirpollutionPredictor import AirpollutionPredictor

import pandas as pd
import numpy as np

import os
from glob import glob

import pickle

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches


base_path = "D:\Programming\Python\SustainabilityReport/"
pollutants = ['NO2', 'O3', 'PM10', 'PM25', 'SO2']
who_particle_limits = {'NO2': 40, 'O3': 100, 'PM10': 20, 'PM25': 10, 'SO2': 20}

region = "ES51"
model_name = "Linear Regression"
policy_dict = {"vreg_CAR": -0.5}

historic_path = glob(os.path.join(base_path, "historic", "{}_historic_*.csv".format(region)))[0]
forecast_path = glob(os.path.join(base_path, "forecast", "{}_forecast_*_data.csv".format(region)))[0]

df_historic = pd.read_csv(historic_path)
df_historic = df_historic.drop(["NUTS_ID", "NUTS_NAME"], axis=1)
df_forecast = pd.read_csv(forecast_path)
df_forecast = df_forecast.drop(["NUTS_ID", "NUTS_NAME"], axis=1)

model_dict = {}
for pollutant in pollutants:
    model_dict[pollutant] = pickle.load(open(os.path.join(base_path, "model", "model_{}_{}.pkl".format(pollutant, model_name)), 'rb'))
columnOrder = pd.read_csv(os.path.join(base_path, "model", "ColumnsInOrder.csv"))["column"].tolist()

airpollutionpredictor = AirpollutionPredictor(pollutants= pollutants, particle_limits= who_particle_limits)
airpollutionpredictor.init_data(df_historic, df_forecast)
airpollutionpredictor.init_model(models= model_dict, columnOrder= columnOrder)
airpollutionpredictor.predict_airpollution(policy_dict)

df = airpollutionpredictor.return_airpollution()

#without confidence intervall
df[pollutants].plot()
plt.title("Devlopment air pollution 1990 to 2030 - restrict traffic")
plt.savefig(os.path.join(base_path, "plots", "airPollution19902030_restTraffic.png"))
plt.show()

# create graph which shows level above the threshold in red
cmap = ["C0", "C1", "C2", "C4", "C6", "C7", "C8", "C9"]
c_above_th = "C3"
for i, pollutant in enumerate(pollutants):
    temp = df.filter(regex=pollutant)
    x = np.array(temp.index)
    y = temp.loc[:, pollutant].values

    thres_passes = np.where((y[~np.isnan(y)]>who_particle_limits[pollutant])[:-1] != (y[~np.isnan(y)]>who_particle_limits[pollutant])[1:])[0]
    for thres_pass in thres_passes:
        slope = 1/ (y[~np.isnan(y)][thres_pass+1] - y[~np.isnan(y)][thres_pass])
        x_new = slope * (who_particle_limits[pollutant] - y[~np.isnan(y)][thres_pass]) + x[~np.isnan(y)][thres_pass]
        x = np.append(x, x_new)
        y = np.append(y, who_particle_limits[pollutant]+ 1/(10**5) * slope)
    x_inds = x.argsort()
    x = x[x_inds]
    y = y[x_inds]
    colormap = ListedColormap([cmap[i], c_above_th])
    highest_value = max(np.nanmax(y), who_particle_limits[pollutant])
    norm = BoundaryNorm([0, who_particle_limits[pollutant], highest_value], colormap.N)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=colormap, norm=norm)
    lc.set_array(y)
    lc.set_linewidth(2)

    plt.gca().add_collection(lc)
    plt.xlim(x.min(), x.max())
    plt.ylim(0, highest_value*1.1)
    plt.hlines(who_particle_limits[pollutant], x[0], x[-1], colors='k', linestyles='solid', label='WHO threshold')
    plt.title("Development of {} (with restricted traffic)".format(pollutant))
    plt.xlabel('year')
    plt.ylabel('µg/m³')
    handles, labels = plt.gca().get_legend_handles_labels()
    patch = mpatches.Patch(color=cmap[i], label=pollutant)
    handles.insert(0, patch)
    plt.legend(handles=handles)
    plt.savefig(os.path.join(base_path, "plots", "Development_{}_{:.0f}_{:.0f}_restTraffic.png".format(pollutant, x.min(), x.max())))
    plt.show()