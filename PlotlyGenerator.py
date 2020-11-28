import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot
from plotly.subplots import make_subplots

class AirpollutionPlotter():
    def __init__(self, colors = None):
        if colors:
            self.colors = colors
        else:
            self.colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    def plot(self, df: pd.DataFrame, who_particle_limits: dict, pollutants: list) -> go.Figure:
        # create graph which shows level above the threshold in red
        fig = make_subplots(rows=len(pollutants), cols=1, shared_xaxes=True, vertical_spacing=0.05)

        for i, pollutant in enumerate(pollutants):
            temp = df.filter(regex=pollutant)
            x = np.array(temp.index)
            y = temp.loc[:, pollutant].values

            thres_passes = np.where((y[~np.isnan(y)]>who_particle_limits[pollutant])[:-1] != (y[~np.isnan(y)]>who_particle_limits[pollutant])[1:])[0]
            for thres_pass in thres_passes:
                slope = 1/ (y[~np.isnan(y)][thres_pass+1] - y[~np.isnan(y)][thres_pass])
                x_new = slope * (who_particle_limits[pollutant] - y[~np.isnan(y)][thres_pass]) + x[~np.isnan(y)][thres_pass]
                x = np.append(x, x_new)
                y = np.append(y, who_particle_limits[pollutant])
            x_inds = x.argsort()
            x = x[x_inds]
            y = y[x_inds]

            #virtual line for legend
            fig.add_trace(go.Scatter(x=x, y=np.full((len(x),), np.nan), line={'color': "rgb({}, {}, {})".format(*self.colors[2*i])}, name=pollutant, legendgroup=pollutant), row=i+1, col=1)
            # Below, above and on threshold
            fig.add_trace(go.Scatter(x=x, y=np.where(y <= who_particle_limits[pollutant], y, np.nan), line={'color': "rgb({}, {}, {})".format(*self.colors[2*i])}, marker={'color': "rgb({}, {}, {})".format(44, 160, 44), 'size': 4}, mode = 'lines+markers', name="{} 🙂".format(pollutant), legendgroup=pollutant, showlegend=False), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=x, y=np.where(y >= who_particle_limits[pollutant], y, np.nan), line={'color': "rgb({}, {}, {})".format(*self.colors[2*i])}, marker={'color': "rgb({}, {}, {})".format(214, 39, 40), 'size': 4}, mode = 'lines+markers', name="{} 💀".format(pollutant), legendgroup=pollutant, showlegend=False), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=x, y=np.where(y == who_particle_limits[pollutant], y, np.nan), marker={'color': 'black', 'size': 4}, mode='markers', name="{}".format(pollutant), legendgroup=pollutant, showlegend=False), row=i+1, col=1)

            #threshold line
            if i == len(pollutants) - 1:
                fig.add_trace(go.Scatter(x=x, y=np.full((len(x),), who_particle_limits[pollutant]), line={'color': 'black', 'width': 1}, name="WHO guidline", legendgroup="whoGuidline"), row=i + 1, col=1)
            else:
                fig.add_trace(go.Scatter(x=x, y=np.full((len(x),), who_particle_limits[pollutant]), line={'color': 'black', 'width': 1}, name="WHO guidline", legendgroup="whoGuidline", showlegend=False), row=i+1, col=1)

            #add infos to the plot
            fig.update_yaxes(title_text="µg/m³", row=i+1, col=1)

            if i == len(pollutants)-1:
                fig.update_xaxes(title_text="year", row=i+1, col=1)

        fig.update_layout(plot_bgcolor= "rgba(0, 0, 0, 0)", paper_bgcolor= "rgba(0, 0, 0, 0)")
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, x=0.5, xanchor="center"))
        return fig