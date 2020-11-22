import os
from nuts_finder import NutsFinder
nf = NutsFinder(year=2016)
import pandas as pd
import numpy as np

base_path = r"D:\Programming\Python\SustainabilityReport\data\yearly"
pollutants = ["NO2", "O3", "PM10", "PM25", "SO2"]

def get_NutsCode(row: pd.Series) -> pd.Series:
    try:
        result = nf.find(lat=row["SamplingPoint_Latitude"], lon=row["SamplingPoint_Longitude"])
        levels = [r['LEVL_CODE'] for r in result]
        result = result[levels.index(2)]
        return pd.Series({'NUTS_ID': result['NUTS_ID'], 'NUTS_NAME': result['NUTS_NAME']})
    except:
        return pd.Series({'NUTS_ID': np.nan, 'NUTS_NAME': np.nan})

for pollutant in pollutants:
    print(pollutant)
    df = pd.read_csv(os.path.join(base_path, "airpollution\\mean", "{}_mean.csv".format(pollutant)))
    df = pd.concat([df, df.apply(get_NutsCode, axis= 1)], axis=1, sort=False)
    df = df.dropna(subset= ["NUTS_ID", "NUTS_NAME"])
    df = df.groupby(["NUTS_ID", "NUTS_NAME", "ReportingYear"])["AQValue"].mean().reset_index()
    df.to_csv(os.path.join(base_path, "airpollution\\mean", "{}_mean_NUTS2_2016.csv".format(pollutant)), index= False)