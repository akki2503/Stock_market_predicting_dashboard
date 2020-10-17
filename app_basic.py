#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd


# In[5]:
df = pd.read_csv('INFY.NS.csv', index_col=0, parse_dates=True)

import plotly.express as px

# In[18]:
mgr_options = list(df.columns)

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
                value='Close'),
        ],
        style={'width': '25%',
               'display': 'inline-block'}),
    dcc.Graph(id='funnel-graph'),
])


@app.callback(
    dash.dependencies.Output('funnel-graph', 'figure'),
    [dash.dependencies.Input('Stock', 'value')])
def update_graph(Stock):
    if Stock == "Close":
        df_plot = df.copy()
    else:
        df_plot = df.copy()
    
    fig = go.Scatter(x=df_plot.index, y=df[Stock])
    # fig = go.figure(px.line(df_plot, x=df_plot.index, y=df[Stock]))

    return {
        'data': [fig],
        'layout':
            go.Layout(title='Stock Close Price')
    }
if __name__ == '__main__':
    app.run_server(debug=True)
