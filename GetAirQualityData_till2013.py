import pandas as pd
import numpy as np
import requests
import glob
import os
import airbase
client = airbase.AirbaseClient()
from nuts_finder import NutsFinder
nf = NutsFinder(year=2016)

def get_NutsCode(row: pd.Series) -> pd.Series:
    try:
        result = nf.find(lat=row["station_latitude_deg"], lon=row["station_longitude_deg"])
        levels = [r['LEVL_CODE'] for r in result]
        result = result[levels.index(2)]
        return [result['NUTS_ID'], result['NUTS_NAME']]
    except:
        return [np.nan, np.nan]

def translate_stationCode(row: pd.Series) -> pd.Series:
    try:
        temp = airstations_nuts2_dict[row['station_european_code']]
        row['NUTS_ID'] = temp[0]
        row['NUTS_NAME'] = temp[1]
    except:
        row['NUTS_ID'] = np.nan
        row['NUTS_NAME'] = np.nan
    return row

default_path = r"D:\Programming\Python\SustainabilityReport\data/"
pollutants = ['Nitrogen dioxide (air)', 'Ozone (air)', 'Particulate matter < 10 µm (aerosol)', 'Particulate matter < 2.5 µm (aerosol)', 'Sulphur dioxide (air)']
pollutant_dict = {'Nitrogen dioxide (air)': 'NO2',
                  'Ozone (air)': 'O3',
                  'Particulate matter < 10 µm (aerosol)': 'PM10',
                  'Particulate matter < 2.5 µm (aerosol)': 'PM25',
                  'Sulphur dioxide (air)': 'SO2'
                  }

airstations = pd.read_csv(default_path + r"airbase/AirBase_v8_stations.csv", sep="\t")
airstations = airstations.loc[:, ["station_european_code", "station_longitude_deg", "station_latitude_deg", "station_altitude"]]
airstations = airstations.groupby(by= "station_european_code").max().reset_index()
#airstations = airstations.iloc[:10, :]
airstations["NUTS_2"] = airstations.apply(get_NutsCode, axis=1)
airstations = airstations.loc[~airstations["NUTS_2"].apply(lambda x: x[0]).isna()]
airstations_nuts2_dict = dict(zip(airstations["station_european_code"], airstations["NUTS_2"]))

start_year = 1990
end_year = 2012
df = pd.read_csv(default_path + r"airbase/AirBase_v8_statistics.csv", sep="\t")
df = df.loc[df['component_name'].isin(pollutants) & (df['statistic_shortname'] == "Mean"), ['station_european_code', 'component_name', 'statistics_year', 'statistic_value']]
df.replace({'component_name': pollutant_dict})
df = df.loc[df['statistics_year'].isin(list(range(start_year, end_year)))]
df = df.apply(translate_stationCode, axis= 1)
df = df.groupby(by= ['NUTS_ID', 'NUTS_NAME', 'component_name', 'statistics_year']).mean().reset_index()

inv_pollutant_dict = {v: k for k, v in pollutant_dict.items()}

for pollutant in map(pollutant_dict.get, pollutants):
    print(pollutant)
    temp_df = df.copy()
    temp_old = temp_df.loc[temp_df["component_name"] == inv_pollutant_dict[pollutant]]
    temp_old = temp_old.loc[:, ['NUTS_ID', 'NUTS_NAME', 'statistics_year', 'statistic_value']]
    temp_old.columns = ['NUTS_ID', 'NUTS_NAME', 'ReportingYear', 'AQValue']
    temp_new = pd.read_csv(default_path + r"yearly\airpollution\mean\{}_mean_NUTS2_2016.csv".format(pollutant))
    temp_df = pd.concat([temp_new, temp_old], ignore_index=True)
    temp_df.to_csv(default_path + r"yearly\airpollution\mean\{}_mean_NUTS2_2016_full.csv".format(pollutant), index= False)