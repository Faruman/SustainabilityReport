import pandas as pd
import numpy as np

class Nuts2Converter:
    def __init__(self, path):
        load_dict = dict()
        self.nuts2_dict_list = list()
        self.year_list = list()

        xlsx = pd.ExcelFile(r'D:\Programming\Python\SustainabilityReport\data\general\NUTS2_conversion.xlsx')
        for sheet in xlsx.sheet_names:
            load_dict[sheet[:4]] = pd.read_excel(xlsx, sheet_name=sheet)

        keys = list(load_dict.keys())
        keys.sort(reverse= True)

        for key in keys:
            df = load_dict[key]
            for column in df.columns:
                df = df.loc[df[column].str.len() == 4]
            if len(self.nuts2_dict_list) > 0:
                nuts2_dict = self.nuts2_dict_list[-1]
                df.replace({column: nuts2_dict})
            df = df.dropna()
            df.columns = [key, "2016"]
            self.nuts2_dict_list.append(dict(zip(list(df.iloc[:, 0].values), list(df.iloc[:, -1].values))))
            self.year_list.append(key)
    def convert(self, str):
        return_str = np.nan
        for i, nuts_dict in enumerate(self.nuts2_dict_list):
            if i == 0:
                if str in nuts_dict.values():
                    return_str = str
                    break
            if str in nuts_dict.keys():
                return_str = nuts_dict[str]
                break
        return return_str

MyConverter = Nuts2Converter(r"D:\Programming\Python\SustainabilityReport\data\general\NUTS2_conversion.xlsx")
MyConverter.convert("DE41")