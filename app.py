import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

model_tcs = tf.keras.models.load_model('/Users/abhyudoysmac/Desktop/Machine Learning/models/model_TCS.h5')
model_hdfc = tf.keras.models.load_model('/Users/abhyudoysmac/Desktop/Machine Learning/models/model_HDFC.h5')
model_bhartartl = tf.keras.models.load_model('/Users/abhyudoysmac/Desktop/Machine Learning/models/model_BHARTIARTL.h5')
model_reliance = tf.keras.models.load_model('/Users/abhyudoysmac/Desktop/Machine Learning/models/model_RELIANCE.h5')
model_hcltech = tf.keras.models.load_model('/Users/abhyudoysmac/Desktop/Machine Learning/models/model_HCLTECH.h5')

def load_stock_data(stock):
    file_name = f"data/{stock}.csv"
    df = pd.read_csv(file_name)
    return df

def preprocess_data(df):
    features = ['Open', 'High', 'Low', 'Volume']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])
    X = pd.DataFrame(columns=features, data=X, index=df.index)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X

app = dash.Dash(__name__, external_stylesheets=['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'])

app.layout = html.Div([
    html.Div(
        children=[
            "Made with",
            html.I(className='fa fa-heart', style={'color': 'red', 'fontSize': 24, 'marginRight': 10}),
            "by Abhyudoy Chaki and Shiva Khanduri"
        ],
        style={'textAlign': 'center', 'fontSize': 16, 'marginTop': 10}
    ),
    html.H1("Stock Price VWAP Prediction using NIFTY50 Stock Data"),
    dcc.Dropdown(
        id='stock-dropdown',
        options=[
            {'label': 'TCS', 'value': 'TCS'},
            {'label': 'HDFC', 'value': 'HDFC'},
            {'label': 'BHARTIARTL', 'value': 'BHARTIARTL'},
            {'label': 'RELIANCE', 'value': 'RELIANCE'},
            {'label': 'HCLTECH', 'value': 'HCLTECH'}
        ],
        value='HCLTECH',
        style={'width': '50%'}
    ),
    dcc.RangeSlider(
        id='date-slider',
        min=0,
        max=1000,
        step=1,
        marks={i: f'Day {i}' for i in range(0, 1001, 100)},
        value=[0, 1000]
    ),
    dcc.Graph(id='stock-graph'),
    html.Div(id='raw-data-table'),
    html.Button('Reset Date Range', id='reset-button', n_clicks=0, style={'display': 'block', 'margin': '20px auto'})
])

@app.callback(
    [Output('stock-graph', 'figure'),
     Output('raw-data-table', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('date-slider', 'value'),
     Input('reset-button', 'n_clicks')]
)
def update_graph(selected_stock, selected_dates, reset_clicks):
    if reset_clicks > 0:
        selected_dates = [0, 1000]

    df = load_stock_data(selected_stock)
    start_date, end_date = selected_dates
    df = df.iloc[start_date:end_date]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    X = preprocess_data(df)

    if selected_stock == 'TCS':
        model = model_tcs
    elif selected_stock == 'HDFC':
        model = model_hdfc
    elif selected_stock == 'BHARTIARTL':
        model = model_bhartartl
    elif selected_stock == 'RELIANCE':
        model = model_reliance
    else:
        model = model_hcltech

    y_pred = model.predict(X)

    trace1 = go.Scatter(
        x=df['Date'],
        y=df['VWAP'],
        mode='lines',
        name='True VWAP'
    )
    trace2 = go.Scatter(
        x=df['Date'],
        y=y_pred.flatten(),
        mode='lines',
        name='Predicted VWAP'
    )

    figure = {
        'data': [trace1, trace2],
        'layout': go.Layout(
            title=f'{selected_stock} Stock VWAP Prediction',
            xaxis={'title': 'Date'},
            yaxis={'title': 'VWAP'}
        )
    }

    features = ['Open', 'High', 'Low', 'Volume', 'VWAP']
    raw_data_table = html.Div(
        children=html.Table([
            html.Tr([html.Th(col) for col in features])
        ] + [
            html.Tr([html.Td(df.iloc[i][col]) for col in features])
            for i in range(len(df))
        ]),
        style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'text-align': 'center'}
    )

    return figure, raw_data_table

if __name__ == '__main__':
    app.run_server(debug=True)
