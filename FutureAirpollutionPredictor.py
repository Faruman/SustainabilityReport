import pandas as pd
import pickle
import glob
import os

class AirpollutionPredictor:
    def __init__(self, pollutants, particle_limits):
        self.pollutants = pollutants
        self.particle_limits = particle_limits
        self.hist_airp = pd.DataFrame()
        self.frcst_airp = pd.DataFrame()
        self.frcst_data = pd.DataFrame()
        self.models_airp = dict()

    def init_data(self, historic_air_pollution_data: pd.DataFrame, forecast_covariates: pd.DataFrame):
        self.hist_airp = historic_air_pollution_data
        self.hist_airp = self.hist_airp.set_index("year")
        self.frcst_data = forecast_covariates
        self.frcst_data = self.frcst_data.set_index("year")

    def init_model(self, models: dict, columnOrder: list):
        if not set(models.keys()) - set(self.pollutants):
            self.models_airp = models
            self.models_airp_columnOrder = columnOrder
        else:
            raise Exception("Too few Models found. Models for all Pollutants need to be provided.")

    def predict_airpollution(self, policy_dict:dict= dict()):
        if not self.models_airp:
            raise Exception("No model found. Please use .init_model to load the model first")
        elif self.hist_airp.empty or self.frcst_data.empty:
            raise Exception("No data found. Please use .init_data to load the model first")
        else:
            frcst_data = self.frcst_data.copy(deep=True)
            for key in policy_dict.keys():
                filter_columns = [s for s in frcst_data.columns if key in s]
                base = frcst_data.loc[frcst_data.index.min(), filter_columns]
                forecast = base + (base * policy_dict[key])
                slope = (forecast - base) / (frcst_data.index.max() - frcst_data.index.min())
                for year in list(set(frcst_data.index) - set([frcst_data.index.min()])):
                    frcst_data.loc[year, filter_columns] = base + slope * (year - frcst_data.index.min())

            frcst_airp = pd.DataFrame(index=frcst_data.index, columns=self.pollutants)

            for pollutant in self.pollutants:
                model = self.models_airp[pollutant]
                selected_columns = [s for s in frcst_data.columns if "yhat_" in s]
                df_temp = frcst_data[selected_columns]
                df_temp.columns = df_temp.columns.str.replace("yhat_", '')
                df_temp = df_temp[self.models_airp_columnOrder]
                y_pred = model.predict(df_temp)
                frcst_airp.loc[:, pollutant] = y_pred
            self.frcst_airp = frcst_airp
            del frcst_data

    def return_airpollution_forecast(self):
        return self.frcst_airp

    def return_airpollution(self):
        return self.hist_airp.append(self.frcst_airp)