#!/usr/bin/env python
# coding: utf-8

# In[10]:

import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from lstm_model_py import forecast_lstm, forecast

## function to read close prices of previous 5 days
def read_stock_data(stock_name):
    stock_data = pd.read_csv(stock_name+'.csv')
    return stock_data

## predictions using prev data
def lstm_prediction(prev_data, lstm_model):
    x_train_l = prev_data.reshape(1,1,prev_data.shape[0])
    x_train_l = np.asarray(x_train_l).astype(np.float32)

    forecasts = list()
    for i in range(len(x_train_l)):
        x= x_train_l[i,:]
        fore = forecast_lstm(lstm_model, x , 1)
        forecasts = np.append(forecasts, fore)

    return forecasts

## load lstm model
def load_model():
    global model
    model = keras.models.load_model('lstm_model')
    return model



import plotly.express as px

mgr_options = ['Infosys']

app = dash.Dash()
app.layout = html.Div([
    html.H2("Stock Prediction Dashboard"),
    html.Div(
        [
            dcc.Dropdown(
                id="Stock",
                options=[{
                    'label': i,
                    'value': i,
                } for i in mgr_options],
                value='Infosys'),
        ],
        style={'width': '25%',
               'display': 'inline-block'}),
    dcc.Graph(id='funnel-graph'),
    dcc.Textarea(
        id='textarea-example',
        value='Textarea content initialized\nwith multiple lines of text',
        style={'width': '50%', 'height': 100},
    ),
])


@app.callback(
    dash.dependencies.Output('funnel-graph', 'figure'),
    [dash.dependencies.Input('Stock', 'value')])
def update_graph(Stock):
    df_plot = read_stock_data(Stock)
    model = load_model()
    preprocessed_data = process_data_for_prediction(df_plot, model)
    df2 = pd.DataFrame(preprocessed_data[:,4], columns=['events'])
    fig = go.Scatter(x=df2.index, y=df2['events'])
    return {
        'data': [fig],
        'layout':
            go.Layout(title='Stock Close Price')
    }
if __name__ == '__main__':
    model = load_model()
    app.run_server(debug=True)

# %%
