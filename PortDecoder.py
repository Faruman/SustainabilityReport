import pandas as pd
import numpy as np
from nuts_finder import NutsFinder
nf = NutsFinder(year=2016)
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="PortDecoder")
import googlemaps
gmaps = googlemaps.Client(key="{my_key}")
import eurostat

def getLatAndLon(row):
    direction_dict = {'N': 1, 'S': -1, 'E': 1, 'W': -1}
    if not pd.isna(row["Coordinates"]):
        text = row["Coordinates"].split(" ")
        return pd.Series({"Latitude": (int(text[0][:-3]) + int(text[0][-3:-1]) / 60.0 * direction_dict[text[0][-1:]]), "Longitude": (int(text[1][:-3]) + int(text[1][-3:-1]) / 60.0 * direction_dict[text[1][-1:]])})
    else:
        try:
            location = geolocator.geocode(row["Name"] + ", " + row["Country"])
            return pd.Series({"Latitude": location.latitude, "Longitude": location.longitude})
        except:
            try:
                result = gmaps.geocode(row["Name"] + ", " + row["Country"])[0]
                location = result["geometry"]["location"]
                return pd.Series({"Latitude": location.lat, "Longitude": location.lon})
            except:
                return pd.Series({"Latitude": np.nan, "Longitude": np.nan})

def get_NutsCode(row: pd.Series) -> pd.Series:
    try:
        result = nf.find(lat=row["Latitude"], lon=row["Longitude"])
        levels = [r['LEVL_CODE'] for r in result]
        result = result[levels.index(2)]
        return pd.Series({'NUTS_ID': result['NUTS_ID'], 'NUTS_NAME': result['NUTS_NAME']})
    except:
        return pd.Series({'NUTS_ID': np.nan, 'NUTS_NAME': np.nan})

locodes = pd.read_csv("D:\Programming\Python\SustainabilityReport\data\general\locode-list.csv")

ports = eurostat.get_data_df('mar_go_aa')
ports = pd.DataFrame(ports.loc[ports[r"rep_mar\time"].str.contains(r'([a-zA-Z]{5})$'), r"rep_mar\time"])
ports = ports.reset_index(drop=True)
ports["Country"] = ports[r"rep_mar\time"].map(lambda x: x[-5:-3])
ports["Location"] = ports[r"rep_mar\time"].map(lambda x: x[-3:])

ports = ports.merge(locodes, on=["Country", "Location"], how="left")
ports = ports.loc[:, [r"rep_mar\time", "Country", "Location", "Name", "Coordinates"]]

ports = ports.merge(ports.apply(getLatAndLon, axis=1), left_index=True, right_index=True)

ports = ports.merge(ports.apply(get_NutsCode, axis=1), left_index=True, right_index=True)

ports = ports.dropna()

ports.loc[:, [r"rep_mar\time", "NUTS_ID", "NUTS_NAME"]].to_excel("D:\Programming\Python\SustainabilityReport\data\general\PortsDict_NUTS2_2016.xlsx", index= False)


