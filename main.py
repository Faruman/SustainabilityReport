import os

import pandas as pd

import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc

from FutureAirpollutionPredictor import AirpollutionPredictor

from PlotlyGenerator import AirpollutionPlotter

#---------------------------------------------------------------

pollutants = ['NO2', 'O3', 'PM10', 'PM25', 'SO2']
who_particle_limits = {'NO2': 40, 'O3': 100, 'PM10': 20, 'PM25': 10, 'SO2': 20}

region = "ES51"
model_name = "Linear Regression"

historic_path = os.path.join("./", "historic", "{}_historic_1990_2019.csv".format(region))
forecast_path = os.path.join("./", "forecast", "{}_forecast_2020_2030_data.csv".format(region))

df_historic = pd.read_csv(historic_path)
df_historic = df_historic.drop(["NUTS_ID", "NUTS_NAME"], axis=1)
df_forecast = pd.read_csv(forecast_path)
df_forecast = df_forecast.drop(["NUTS_ID", "NUTS_NAME"], axis=1)

model_dict = {}
for pollutant in pollutants:
    model_dict[pollutant] = pickle.load(open(os.path.join("./", "model", "model_{}_{}.pkl".format(pollutant, model_name)), 'rb'))
columnOrder = pd.read_csv(os.path.join("./", "model", "ColumnsInOrder.csv"))["column"].tolist()

airpollutionpredictor = AirpollutionPredictor(pollutants= pollutants, particle_limits= who_particle_limits)
airpollutionpredictor.init_data(df_historic, df_forecast)
airpollutionpredictor.init_model(models= model_dict, columnOrder= columnOrder)

airpollutionplotter = AirpollutionPlotter()

#---------------------------------------------------------------

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
# try running the app with one of the Bootswatch themes e.g.
# app = dash.Dash(external_stylesheets=[dbc.themes.JOURNAL])
# app = dash.Dash(external_stylesheets=[dbc.themes.SKETCHY])


header = dbc.Row(
    [
        dbc.Col([html.H3("Sustainability Report", className="text-light")], className="col-12 col-sm-9 col-md-10 "),
        dbc.Col(dbc.Button("Learn more", href="https://github.com/Faruman/SustainabilityReport", color="light", className="btn-block"), className="col-12 col-sm-3 col-md-2")
    ],
    className= "px-5 py-4 bg-dark",
)

explenation = dbc.Row(
    [
        dbc.Col([html.P("As it can be assumed that external factors, such as the number of cars in a region, will affect the airpollution in this region. To tackle airpollution policy can be introduced, forcing or nudging people to change their behaviour. These legislation will therefore change the underlying factors of air pollution. As part of our sustainability report, we build a model which allows us to see how these policy changes affect airpollution levels. By changing the sliders you can determine how much you would like to reduce or increase a certain factor by 2030. If no policy is set the predicted development of the factor will be used.", className="text-dark align-middle text-center")], className="col-12")
    ],
    className= "p-5 bg-light",
)

filter = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Transportation", className="card-title"),
                        html.Div([
                        html.H6("Registered Cars", className="card-subtitle"),
                        html.P("Car stock in the area.",className="card-text",),
                        dcc.Slider(id='vreg_CAR', min=-0.99, max=0.99, step=0.05,value=0,marks={-0.99: '-100%', -0.5: '-50%', 0: '0', 0.5: '+50%',0.99: '+100%'}),
                        ]),
                        html.Div([
                        html.H6("Registered Motorcycles", className="card-subtitle"),
                        html.P("Motorcycle stock in the area.",className="card-text",),
                        dcc.Slider(id='vreg_MOTO', min=-0.99, max=0.99, step=0.05,value=0,marks={-0.99: '-100%', -0.5: '-50%', 0: '0', 0.5: '+50%',0.99: '+100%'})
                        ], className= "mt-1"),
                    ]
                ),
                className= "h-100"
            ),
            className="col-12 col-md-4",
            style={"paddingTop": "15px", "paddingBottom": "15px"}
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Agriculture", className="card-title"),
                        html.Div([
                        html.H6("Livestock", className="card-subtitle"),
                        html.P("Number of production animals.",className="card-text",),
                        dcc.Slider(id='lst_A2000', min=-0.99, max=0.99, step=0.05,value=0,marks={-0.99: '-100%', -0.5: '-50%', 0: '0', 0.5: '+50%',0.99: '+100%'}),
                        ]),
                        html.Div([
                        html.H6("Crop Production", className="card-subtitle"),
                        html.P("Amount of farmland for crops production.",className="card-text",),
                        dcc.Slider(id='crpp_C0000', min=-0.99, max=0.99, step=0.05,value=0,marks={-0.99: '-100%', -0.5: '-50%', 0: '0', 0.5: '+50%',0.99: '+100%'}),
                        ], className="mt-1"),
                    ]
                ),
                className= "h-100"
            ),
            className="col-12 col-md-4",
            style={"paddingTop": "15px", "paddingBottom": "15px"}
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Industry", className="card-title"),
                        html.Div([
                            html.H6("Mining and Quarying", className="card-subtitle"),
                            html.P("Area used for mining and quarying.", className="card-text", ),
                            dcc.Slider(id='ind_B', min=-0.99, max=0.99, step=0.05, value=0,
                                       marks={-0.99: '-100%', -0.5: '-50%', 0: '0', 0.5: '+50%',0.99: '+100%'}),
                        ]),
                        html.Div([
                            html.H6("Manufacturing", className="card-subtitle"),
                            html.P("Area used by the manufacturing industry.", className="card-text", ),
                            dcc.Slider(id='ind_C', min=-0.99, max=0.99, step=0.05, value=0, marks={-0.99: '-100%', -0.5: '-50%', 0: '0', 0.5: '+50%',0.99: '+100%'}),
                        ], className="mt-1"),
                    ]
                ),
                className= "h-100"
            ),
            className="col-12 col-md-4",
            style={"paddingTop": "15px", "paddingBottom": "15px"}
        ),
    ],
    className= "px-5 py-4 bg-dark",
)

MainPlot = dbc.Row([
    dbc.Container([
        dcc.Graph(id='pollution_graph', className="w-100", style={'width': '90vh', 'height': '90vh'})
    ])
    ],
    className= "px-5 py-4 bg-light",
)

app.layout = html.Div(
    [header, explenation, filter, MainPlot]
)

#---------------------------------------------------------------

@app.callback(
    Output('pollution_graph','figure'),
    [Input('vreg_CAR','value'), Input('vreg_MOTO','value'), Input('lst_A2000','value'), Input('crpp_C0000','value'), Input('ind_B','value'), Input('ind_C','value')]
)

def build_graph(vreg_CAR, vreg_MOTO, lst_A2000, crpp_C0000, ind_B, ind_C):
    var_name = ["vreg_CAR", "vreg_MOTO", "lst_A2000", "crpp_C0000", "ind_B", "ind_C"]

    policy_dict = {}

    for i, var in enumerate([vreg_CAR, vreg_MOTO, lst_A2000, crpp_C0000, ind_B, ind_C]):
        if var < 1:
            policy_dict[var_name[i]] = var

    airpollutionpredictor.predict_airpollution(policy_dict)
    df = airpollutionpredictor.return_airpollution()

    fig = airpollutionplotter.plot(df, who_particle_limits, pollutants)

    return fig

#---------------------------------------------------------------

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8080)