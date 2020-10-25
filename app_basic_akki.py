<<<<<<< HEAD:app_basic_akki.py
#!/usr/bin/env python
# coding: utf-8

# In[10]:

import numpy as np
=======
>>>>>>> b9cdd72ba5f00a7cd85f6b34ec7a7f424b759b2c:app_basic.py
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

<<<<<<< HEAD:app_basic_akki.py
from lstm_model_py import forecast_lstm, forecast

## function to read close prices of previous 5 days
=======

>>>>>>> b9cdd72ba5f00a7cd85f6b34ec7a7f424b759b2c:app_basic.py
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

<<<<<<< HEAD:app_basic_akki.py
    return forecasts

## load lstm model
=======
    y_pred = model.predict(x_train_cnn)
    y_pred = sc_y.inverse_transform(y_pred)

    return y_pred

>>>>>>> b9cdd72ba5f00a7cd85f6b34ec7a7f424b759b2c:app_basic.py
def load_model():
    global model
    model = keras.models.load_model('lstm_model')
    return model



import plotly.express as px

<<<<<<< HEAD:app_basic_akki.py
mgr_options = ['Infosys']
=======
mgr_options = ['Infosys'] # contains name used into the prediction model
mgr_dic = {'Infosys': 'infosys-ltd'} # mapping with the webscrapping title
>>>>>>> b9cdd72ba5f00a7cd85f6b34ec7a7f424b759b2c:app_basic.py

import dash_table
import pandas as pd

################################### Web scrapping ##################################

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup
url = "https://trendlyne.com/equity/630/INFY/{0}/".format(mgr_dic[mgr_options[0]])
req = Request(url , headers={'User-Agent': 'Mozilla/5.0'})

webpage = urlopen(req).read()
page_soup = soup(webpage, "html.parser")
import re
analyst = page_soup.find("div", class_ = "stock-metrics")
res = re.findall(r'\{.*?\}',str(analyst))
attri = dict() 
for seg in range(0, 10):
    seg1 = dict(map(lambda x: x.split(':'), res[seg][1:-1].replace("&quot;", "").split(',')))
    seg1 = {x.strip() : y.strip() for x, y in seg1.items()}
    attri[seg1['title']] = seg1['value']
for i in range(10, 18):
    s = res[i][1:-1].replace("&quot;", "").split(',')
    s.remove(s[8])
    s = dict(map(lambda x: x.split(':'), s ))
    s = {x.strip() : y.strip() for x, y in s.items()}
    attri[s['title']] = s['value']

#####################################Web scrapping#####################################

#Financial Statistics Display/ elements

keys = []
vals = []

for i, j in attri.items(): #attri dict contains all the metrics 
    keys.append(i)
    vals.append(j)

#Cards for financial statistics
a1 = dbc.Card(dbc.CardBody([html.H5(str(keys[0]), className="card-title"),html.P(str(vals[0]))]))
a2 = dbc.Card(dbc.CardBody([html.H5(str(keys[1]), className="card-title"),html.P(str(vals[1]))]))
a3 = dbc.Card(dbc.CardBody([html.H5(str(keys[2]), className="card-title"),html.P(str(vals[2]))]))
a4 = dbc.Card(dbc.CardBody([html.H5(str(keys[3]), className="card-title"),html.P(str(vals[3]))]))
a5 = dbc.Card(dbc.CardBody([html.H5(str(keys[4]), className="card-title"),html.P(str(vals[4]))]))
a6 = dbc.Card(dbc.CardBody([html.H5(str(keys[5]), className="card-title"),html.P(str(vals[5]))]))
a7 = dbc.Card(dbc.CardBody([html.H5(str(keys[6]), className="card-title"),html.P(str(vals[6]))]))
a8 = dbc.Card(dbc.CardBody([html.H5(str(keys[7]), className="card-title"),html.P(str(vals[7]))]))
a9 = dbc.Card(dbc.CardBody([html.H5(str(keys[8]), className="card-title"),html.P(str(vals[8]))]))
a10 = dbc.Card(dbc.CardBody([html.H5(str(keys[9]), className="card-title"),html.P(str(vals[9]))]))
a11 = dbc.Card(dbc.CardBody([html.H5(str(keys[10]), className="card-title"),html.P(str(vals[10]))]))
a12 = dbc.Card(dbc.CardBody([html.H5(str(keys[11]), className="card-title"),html.P(str(vals[11]))]))
a13 = dbc.Card(dbc.CardBody([html.H5(str(keys[12]), className="card-title"),html.P(str(vals[12]))]))
a14 = dbc.Card(dbc.CardBody([html.H5(str(keys[13]), className="card-title"),html.P(str(vals[13]))]))


card1 = dbc.Row([dbc.Col(a1, width=4), dbc.Col(a2, width=4), dbc.Col(a3, width=4)])
card2 = dbc.Row([dbc.Col(a4, width=4), dbc.Col(a5, width=4), dbc.Col(a6, width=4)])
card3 = dbc.Row([dbc.Col(a7, width=4), dbc.Col(a8, width=4), dbc.Col(a9, width=4)])
card4 = dbc.Row([dbc.Col(a10, width=4), dbc.Col(a11, width=4), dbc.Col(a12, width=4), dbc.Col(a13, width=4)])

#Graph dash html layout
graph = html.Div(
        [
            dcc.Dropdown(
                id="Stock",
                options=[{
                    'label': i,
                    'value': i,
                } for i, j in mgr_dic.items()],
                value='Infosys'),
        ],
        style={'width': '25%',
               'display': 'inline-block'})

#HMTL elements layout
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    html.H2("Stock Prediction Dashboard"),
    graph,
    dcc.Graph(id='funnel-graph'),
    html.H2("Financial Statistics"),
    card1, card2, card3, card4 
],
)






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
<<<<<<< HEAD:app_basic_akki.py
    model = load_model()
    app.run_server(debug=True)

# %%
=======
    load_model()
    app.run_server("0.0.0.0", 8080,debug=True)
>>>>>>> b9cdd72ba5f00a7cd85f6b34ec7a7f424b759b2c:app_basic.py
