import pandas as pd
import numpy as np
import glob
import os
import eurostat
from sklearn.linear_model import LinearRegression
from CreateNUTS2Converter import Nuts2Converter
from scipy.optimize import curve_fit

base_path = "D:\Programming\Python\SustainabilityReport\data\yearly"
portsDict = pd.read_excel("D:\Programming\Python\SustainabilityReport\data\general\PortsDict_NUTS2_2016.xlsx")
nuts_2_Conv = Nuts2Converter(r"D:\Programming\Python\SustainabilityReport\data\general\NUTS2_conversion.xlsx")

# load the air pollution data and create one data set
airpollution_paths = glob.glob(os.path.join(base_path, "airpollution\mean", "*_mean_NUTS2_2016_full.csv"))
#airpollution_paths = glob.glob(os.path.join(base_path, "airpollution\mean", "*_mean_NUTS2_2016.csv"))

for i, airpollution_path in enumerate(airpollution_paths):
    if i == 0:
        df_pollution = pd.read_csv(airpollution_path)
        df_pollution.columns = [os.path.basename(airpollution_path).split("_")[0] if x=="AQValue" else x for x in df_pollution.columns]
    else:
        temp = pd.read_csv(airpollution_path)
        temp.columns = [os.path.basename(airpollution_path).split("_")[0] if x == "AQValue" else x for x in temp.columns]
        df_pollution = pd.merge(df_pollution, temp, how="outer", on=["NUTS_ID", "NUTS_NAME", "ReportingYear"])
#custom fix for pollutions
customNuts2_dict = {"HR02": "HR04", "HR06": "HR04", "HR06": "HR04","NO0A": "NO05", "UKM8": "UKM3", "UKM9": "UKM3"}
df_pollution.rename({r"NUTS_ID": customNuts2_dict})
df = df_pollution.groupby(by= [r"NUTS_ID", "NUTS_NAME", "ReportingYear"]).mean().reset_index()

time_frame = list(df_pollution["ReportingYear"].unique())
nuts2_regions = list(df_pollution["NUTS_ID"].unique())
# Load the explaining variables
def create_missing_years_mean(row, years):
    average_size = np.mean([row[s] for s in list(row.keys()) if str(s).isdigit()])
    return pd.Series(dict(zip(years, [average_size]*len(years))))

def replace_nan_with_zero(row: pd.Series, exclude_na_rows) -> pd.Series:
    relevant_rows = list(set(row.keys()) - set(["year", r"geo\time"]))
    relevant_values = row[relevant_rows]
    if ((relevant_values.isnull().sum()/ len(relevant_values)) < (1 - exclude_na_rows)) and ((relevant_values.isnull().sum()/ len(relevant_values)) > 0):
        return row.fillna(0)
    else:
        return row

def func_exp(x, a, b, c):
    # c = 0
    return a * np.exp(b * x) + c

class ExponentialRegression:
    def __init__(self, function= func_exp):
        self.func = function
    def fit(self, x, y):
        self.popt, self.pcov = curve_fit(self.func, x, y, p0 = (-1, 0.01, 1))
        return self
    def predict(self, x):
        return self.func(x, *self.popt)

def create_missing_values_Proxy(row, linear_proxy_threshold, method="linear"):
    if (row.isnull().sum() != 0) and ((row.isnull().sum()/ (len(row)-1))  < (1-linear_proxy_threshold)) and ((row.isnull().sum()/ (len(row)-1)) > 0):
        available_years = [s for s in list(row[~ row.isnull()].keys()) if str(s).isdigit()]
        missing_years = [s for s in list(row[row.isnull()].keys()) if str(s).isdigit()]
        available_years.sort()
        available_values = np.array([row[s] for s in available_years])
        base_year = available_years[0]
        available_years_relative = np.array(available_years) - base_year
        missing_years_relative = np.array(missing_years) - base_year
        if method == "expo":
            try:
                reg = ExponentialRegression().fit(available_years_relative, available_values)
                missing_values = []
                for missing_year_relative in missing_years_relative:
                    missing_values.append(reg.predict(missing_year_relative))
                missing_values = np.array(missing_values)
            except:
                reg = LinearRegression().fit(available_years_relative.reshape(-1, 1), available_values.reshape(-1, 1))
                missing_values = reg.predict(missing_years_relative.reshape(-1, 1)).flatten()
        else:
            reg = LinearRegression().fit(available_years_relative.reshape(-1, 1), available_values.reshape(-1, 1))
            missing_values = reg.predict(missing_years_relative.reshape(-1, 1)).flatten()
        missing_values = list(missing_values.reshape(len(missing_values),))
        missing_values = [x if x > 0 else min([i for i in np.append(np.array(available_values), missing_values) if i > 0]) for x in missing_values]
        row[missing_years] = missing_values
        return row
    else:
        return row

def preprocess_eurostat_data(df, time_frame, nuts2_regions, keep_column = None, filter_dict= None, normalisation_frame= pd.DataFrame(), option_replace_nan_with_zero= 0, prefix = None, accumulation_method = "sum", linear_proxy_threshold= 1/3, proxy_method= "linear"):
    if keep_column:
        leading_columns = [r"geo\time"] + [keep_column]
    else:
        leading_columns = [r"geo\time"]
    columns_avail = list(set(time_frame).intersection(set(df.columns)))
    columns_needed = list(set(time_frame).difference(set(df.columns)))
    if filter_dict:
        for filter_name in filter_dict.keys():
            df = df.loc[df[filter_name].isin(filter_dict[filter_name])]
    df.loc[:, r"geo\time"] = df.apply(lambda row: nuts_2_Conv.convert(row[r"geo\time"]), axis=1)
    df = df.loc[df[r"geo\time"].isin(nuts2_regions), leading_columns + columns_avail]
    if accumulation_method == "mean":
        df = df.groupby(by= leading_columns).mean().reset_index()
    else:
        df = df.groupby(by= leading_columns).sum().reset_index()
    df.replace(0, np.nan, inplace=True)
    for column in columns_needed:
        df[column] = np.nan
    df = df.apply(create_missing_values_Proxy, axis=1, args=(linear_proxy_threshold, proxy_method ))
    df = df.melt(id_vars=leading_columns, var_name="year", value_name="value")
    if not normalisation_frame.empty:
        denominator = pd.merge(df[[r"geo\time", "year"]], normalisation_frame, how="left", on=[r"geo\time", "year"])[list(set(normalisation_frame.columns)-set([r'geo\time', 'year']))]
        df["value"] = df[["value"]].values/denominator.values
    if keep_column:
        df = df.pivot(index=[r'geo\time', 'year'], columns=keep_column)['value'].reset_index()
    if option_replace_nan_with_zero == 1:
        df = df.fillna(0)
    elif option_replace_nan_with_zero == 2:
        df = df.apply(replace_nan_with_zero, axis= 1, args=(1/2, ))
    elif option_replace_nan_with_zero == 3:
        df = df.apply(replace_nan_with_zero, axis=1, args=(1/100, ))
    if prefix:
        df = df.add_prefix(prefix)
    return df

## size
filter_landuse = ["TOTAL"]
df_size = eurostat.get_data_df('tgs00002')
columns_avail = list(set(time_frame).intersection(set(df_size.columns)))
columns_needed = list(set(time_frame).difference(set(df_size.columns)))
df_size = df_size.loc[df_size["landuse"].isin(filter_landuse) & df_size[r"geo\time"].isin(nuts2_regions), [r"geo\time"] + columns_avail]
df_size = df_size.merge(df_size.apply(create_missing_years_mean, axis=1, args=(columns_needed, )), left_index=True, right_index=True)
df_size = df_size.melt(id_vars=[r"geo\time"], var_name="year", value_name="size")
df_size["size"] = df_size["size"]/1000

## agriculture
filter_livestock = {"animals": ["A2000"]}
df_livestock = eurostat.get_data_df('agr_r_animal')
df_livestock = preprocess_eurostat_data(df_livestock, time_frame, nuts2_regions, "animals", filter_livestock, normalisation_frame= df_size, option_replace_nan_with_zero= 3, prefix="lst_", linear_proxy_threshold= 1/4)
filter_cropproduction = {"crops": ["C0000", "P0000", "R0000", "I0000", "G0000", "N0000", "E0000"], "strucpro": ["AR"]}
df_cropproduction = eurostat.get_data_df('apro_cpshr')
df_cropproduction = preprocess_eurostat_data(df_cropproduction, time_frame, nuts2_regions, "crops", filter_cropproduction, normalisation_frame= df_size, option_replace_nan_with_zero= 3, prefix="crpp_", linear_proxy_threshold= 1/4)


## heating and cooling
df_heatcool = eurostat.get_data_df('nrg_chddr2_a')
df_heatcool = preprocess_eurostat_data(df_heatcool, time_frame, nuts2_regions, "indic_nrg", prefix= "heco_", accumulation_method="mean", option_replace_nan_with_zero= 1)

## industry
filter_industry = {"nace_r2": ["B", "C", "D", "E", "F", "G", "H", "I", "J", "L", "M", "N"], "indic_sb": ["V11210"]}
df_industry = eurostat.get_data_df('sbs_r_nuts06_r2')
df_industry = preprocess_eurostat_data(df_industry, time_frame, nuts2_regions, "nace_r2", filter_industry, normalisation_frame= df_size, prefix= "ind_", option_replace_nan_with_zero= 2)

## air transportation
filter_air_freight = {"tra_meas": ["FRM_LD_NLD"]}
df_air_freight = eurostat.get_data_df('tran_r_avgo_nm')
df_air_freight = preprocess_eurostat_data(df_air_freight, time_frame, nuts2_regions, filter_dict= filter_air_freight, option_replace_nan_with_zero= 1)
filter_air_person = {"tra_meas": ["PAS_CRD"]}
df_air_person = eurostat.get_data_df('tran_r_avpa_nm')
df_air_person = preprocess_eurostat_data(df_air_person, time_frame, nuts2_regions, filter_dict= filter_air_person, option_replace_nan_with_zero= 1)

## maritime transportation
filter_maritime_weight = {"direct": ["TOTAL"]}
df_maritime_weight = eurostat.get_data_df('mar_go_aa')
df_maritime_weight = pd.merge(df_maritime_weight, portsDict, how='inner', on=r"rep_mar\time")
df_maritime_weight[r"geo\time"] = df_maritime_weight["NUTS_ID"]
df_maritime_weight = df_maritime_weight.groupby([r"geo\time", "direct"]).sum().reset_index()
df_maritime_weight = preprocess_eurostat_data(df_maritime_weight, time_frame, nuts2_regions, filter_dict= filter_maritime_weight, option_replace_nan_with_zero= 1)
filter_maritime_person = {"direct": ["TOTAL"]}
df_maritime_person = eurostat.get_data_df('mar_pa_aa')
df_maritime_person = pd.merge(df_maritime_person, portsDict, how='inner', on=r"rep_mar\time")
df_maritime_person[r"geo\time"] = df_maritime_person["NUTS_ID"]
df_maritime_person = df_maritime_person.groupby([r"geo\time", "direct"]).sum().reset_index()
df_maritime_person = preprocess_eurostat_data(df_maritime_person, time_frame, nuts2_regions, filter_dict= filter_maritime_person, option_replace_nan_with_zero= 1)

## number of cars
filter_vehicles = {"vehicle": ["TOT_X_TM", "TRL_STRL", "MOTO", "CAR"], "unit": ["NR"]}
df_vehicles = eurostat.get_data_df('tran_r_vehst')
df_vehicles = preprocess_eurostat_data(df_vehicles, time_frame, nuts2_regions, "vehicle", filter_vehicles, normalisation_frame= df_size, prefix="vreg_")

## waste
filter_waste = {"indic_env": ["CAP"], "wst_oper": ["DSP_I", "RCV_E"]}
df_waste = eurostat.get_data_df('env_wasfac')
df_waste = preprocess_eurostat_data(df_waste, time_frame, nuts2_regions, "wst_oper", filter_waste, normalisation_frame= df_size, prefix="wst_", option_replace_nan_with_zero= 2, linear_proxy_threshold= 1/8)

## population
df_population = eurostat.get_data_df('tgs00096')
df_population = preprocess_eurostat_data(df_population, time_frame, nuts2_regions, normalisation_frame= df_size, linear_proxy_threshold= 1/8)

#merge everything in the final dataframe
df = pd.merge(df_pollution, df_livestock, how="left", left_on=['NUTS_ID', 'ReportingYear'], right_on=[r"lst_geo\time", 'lst_year'])
df = df.drop([r"lst_geo\time", 'lst_year'], axis=1)
df = pd.merge(df, df_cropproduction, how="left", left_on=['NUTS_ID', 'ReportingYear'], right_on=[r"crpp_geo\time", 'crpp_year'])
df = df.drop([r"crpp_geo\time", 'crpp_year'], axis=1)
df = pd.merge(df, df_heatcool, how="left", left_on=['NUTS_ID', 'ReportingYear'], right_on=[r"heco_geo\time", 'heco_year'])
df = df.drop([r"heco_geo\time", 'heco_year'], axis=1)
df = pd.merge(df, df_industry, how="left", left_on=['NUTS_ID', 'ReportingYear'], right_on=[r"ind_geo\time", 'ind_year'])
df = df.drop([r"ind_geo\time", 'ind_year'], axis=1)
df = pd.merge(df, df_air_freight, how="left", left_on=['NUTS_ID', 'ReportingYear'], right_on=[r"geo\time", 'year'])
df = df.drop([r'geo\time', 'year'], axis=1)
df = df.rename(columns = {'value':'air_freight'})
df = pd.merge(df, df_air_person, how="left", left_on=['NUTS_ID', 'ReportingYear'], right_on=[r"geo\time", 'year'])
df = df.drop([r'geo\time', 'year'], axis=1)
df = df.rename(columns = {'value':'air_person'})
df = pd.merge(df, df_maritime_weight, how="left", left_on=['NUTS_ID', 'ReportingYear'], right_on=[r"geo\time", 'year'])
df = df.drop([r'geo\time', 'year'], axis=1)
df = df.rename(columns = {'value':'marit_freight'})
df["marit_freight"] = df["marit_freight"].fillna(0)
df = pd.merge(df, df_maritime_person, how="left", left_on=['NUTS_ID', 'ReportingYear'], right_on=[r"geo\time", 'year'])
df = df.drop([r'geo\time', 'year'], axis=1)
df = df.rename(columns = {'value':'marit_person'})
df["marit_person"] = df["marit_person"].fillna(0)
df = pd.merge(df, df_vehicles, how="left", left_on=['NUTS_ID', 'ReportingYear'], right_on=[r"vreg_geo\time", 'vreg_year'])
df = df.drop([r"vreg_geo\time", 'vreg_year'], axis=1)
df = pd.merge(df, df_waste, how="left", left_on=['NUTS_ID', 'ReportingYear'], right_on=[r"wst_geo\time", 'wst_year'])
df = df.drop([r"wst_geo\time", 'wst_year'], axis=1)
df = pd.merge(df, df_population, how="left", left_on=['NUTS_ID', 'ReportingYear'], right_on=[r"geo\time", 'year'])
df = df.drop([r'geo\time', 'year'], axis=1)
df = df.rename(columns = {'value':'population'})

df = df.rename(columns= {"ReportingYear": "year"})

df = df.groupby(by=["NUTS_ID", "NUTS_NAME", "year"]).mean().reset_index()

#which data to drop due to missing values?
## var1: only locations with all valid datapoints
df = df.dropna(subset= list(set(df.columns) - set(["NUTS_ID", "NUTS_NAME", "year", "NO2", "O3", "PM10", "PM25", "SO2"])))
## var2:


df.to_csv(os.path.join(base_path[:-6], "data.csv"), index = False)