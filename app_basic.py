#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# In[5]:
def read_stock_data(stock_name):
    stock_data = pd.read_csv(stock_name+'.csv')
    return stock_data

def process_data_for_prediction(raw_data, model):
    n_lag = 5
    raw_data = raw_data.drop(['Adj Close'], axis=1)
    for i in range(1,n_lag+1):
        var_name = 'Shifted Close Price ' + str(i)
        raw_data[var_name] = raw_data['Close'].shift(i)
    raw_data = raw_data.dropna()

    training_set = raw_data.values[-100:]
    training_set.shape

    sc_x = MinMaxScaler(feature_range = (0, 1))
    training_set_x_scaled = sc_x.fit_transform(training_set[:,1:11])
    sc_y = MinMaxScaler(feature_range = (0, 1))
    training_set_y_scaled = sc_y.fit_transform(training_set[:,-1:])

    train_sub_x = training_set_x_scaled[:]
    train_sub_y = training_set_y_scaled[:]
    # test_sub_x = training_set_x_scaled[4500:]
    # test_sub_y = training_set_y_scaled[4500:]

    x_train_cnn = train_sub_x.reshape(train_sub_x.shape[0],train_sub_x.shape[1],1)

    # x_test_cnn = test_sub_x.reshape(test_sub_x.shape[0],test_sub_x.shape[1],1)
    # y_test_cnn = test_sub_y

    y_pred = model.predict(x_train_cnn)
    y_pred = sc_y.inverse_transform(y_pred)

    return y_pred

# def plot(y_pred, raw_data):
#     df2 = pd.DataFrame(y_pred, columns=['events'])
#     fig = go.figure(px.line(df2, x=df2.index, y=df2['events']))
#     return fig

def load_model():
    global model
    model = keras.models.load_model('cnn_model')
    return model

import plotly.express as px

# In[18]:
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
])


@app.callback(
    dash.dependencies.Output('funnel-graph', 'figure'),
    [dash.dependencies.Input('Stock', 'value')])
def update_graph(Stock):
    # if Stock == "Close":
    #     df_plot = df.copy()
    # else:
    #     df_plot = df.copy()
    df_plot = read_stock_data(Stock)
    # plot_values = df_plot['Close']
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
