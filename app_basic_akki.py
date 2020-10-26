import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import pandas_datareader as pdr

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from lstm_model_py import forecast_lstm, forecast
import yfinance as yf
from datetime import date, datetime, timedelta

import dash_table
import pandas as pd
import plotly.express as px

## function to read close prices of previous 5 days
def read_stock_data(stock_name, start_date, end_date):
    stock_data = pdr.get_data_yahoo(stock_name, start_date, end_date)
    data_close = stock_data['Close'].values
    return data_close

## predictions using prev data
def lstm_prediction(prev_data, lstm_model):
    forecasts = forecast_lstm(prev_data, lstm_model)
    return forecasts

def load_model():
    global model
    model = keras.models.load_model('lstm_model')
    return model


# Date Range picker HTML
date_range_picker = html.Div([  dcc.DatePickerSingle(
                                            id='my-date-picker-single',
                                            min_date_allowed=date(2020, 1, 1),
                                            max_date_allowed=date(2030, 1, 1),
                                            initial_visible_month = date.today(),
                                            date = pd.Timestamp(date.today()+ \
                                                             pd.tseries.offsets.BusinessDay(n = -5)).date()
                                            
                                        ),
                                        html.Div(id='output-container-date-picker-single')
                              ])




#HMTL elements layout
app = dash.Dash()
app.layout = html.Div([
    html.H2("Stock Prediction Dashboard"),
    date_range_picker,
    html.Div(
    [
        dcc.Dropdown(
            id="Manager",
            options=[{'label':'INFY.BO', 'value':'INFY.BO'}],
            value='INFY.BO'),
    ],
    style={'width': '25%',
            'display': 'inline-block'}),
    html.Table([
        html.Tr([html.Td(id='day5'), html.Td(id='5'), html.Td(id='5a')]),
        html.Tr([html.Td(id='day4'), html.Td(id='4'), html.Td(id='4a')]),
        html.Tr([html.Td(id='day3'), html.Td(id='3'), html.Td(id='3a')]),
        html.Tr([html.Td(id='day2'), html.Td(id='2'), html.Td(id='2a')]),
        html.Tr([html.Td(id='day1'), html.Td(id='1'), html.Td(id='1a')])
    ])
])


def prev_dates(last_date):
    last_date = pd.Timestamp(last_date)
    bd = pd.tseries.offsets.BusinessDay(n = -4)
    dates = str((last_date+bd).date())
    return dates


def next_dates(last_date):
    last_date = pd.Timestamp(last_date)
    dates = []
    for i in range(5):
        bd = pd.tseries.offsets.BusinessDay(n = i+1)
        dates.append(str((last_date+bd).date()))
    return dates

def act_dates(last_date):
    last_date = pd.Timestamp(last_date)
    bd = pd.tseries.offsets.BusinessDay(n = 5)
    end_dates = str((last_date+bd).date())
    return end_dates

@app.callback(
    [dash.dependencies.Output('5', 'children'),
    dash.dependencies.Output('4', 'children'),
    dash.dependencies.Output('3', 'children'),
    dash.dependencies.Output('2', 'children'),
    dash.dependencies.Output('1', 'children'),
    dash.dependencies.Output('day5', 'children'),
    dash.dependencies.Output('day4', 'children'),
    dash.dependencies.Output('day3', 'children'),
    dash.dependencies.Output('day2', 'children'),
    dash.dependencies.Output('day1', 'children'),
    dash.dependencies.Output('5a', 'children'),
    dash.dependencies.Output('4a', 'children'),
    dash.dependencies.Output('3a', 'children'),
    dash.dependencies.Output('2a', 'children'),
    dash.dependencies.Output('1a', 'children')],
    [dash.dependencies.Input('my-date-picker-single', 'date')])
def predicted_data(last_date):

    prev_date = prev_dates(last_date)    
    data_close = read_stock_data("INFY.BO", prev_date, last_date)
    lstm_model = load_model()
    data_close = data_close.reshape(1,5)
    preprocessed_data = lstm_prediction(data_close, lstm_model)
    df2 = pd.DataFrame(preprocessed_data, columns=['events'], index=next_dates(last_date))
    end_date = act_dates(last_date)
    data_close_act = read_stock_data("INFY.BO", last_date, end_date)
    # fig = go.Scatter(x=df2.index, y=df2['events'])
    return df2['events'].values[0],df2['events'].values[1],df2['events'].values[2], \
           df2['events'].values[3], df2['events'].values[4], df2.index[0], df2.index[1], \
           df2.index[2], df2.index[3], df2.index[4], data_close_act[0], \
           data_close_act[1],data_close_act[2], \
           data_close_act[3], data_close_act[4]


@app.callback(
    dash.dependencies.Output('output-container-date-picker-single', 'children'),
    [dash.dependencies.Input('my-date-picker-single', 'date')])
def update_output(date_value):
    string_prefix = 'You have selected: '
    if date_value is not None:
        date_object = date.fromisoformat(date_value)
        date_string = date_object.strftime('%B %d, %Y')
        return string_prefix + date_string

if __name__ == '__main__':
    app.run_server(debug=True)
