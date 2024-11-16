import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load your models for each stock (assumed to be pre-trained)
model_tcs = tf.keras.models.load_model('/Users/abhyudoysmac/Desktop/Machine Learning/models/model_TCS.h5')
model_hdfc = tf.keras.models.load_model('/Users/abhyudoysmac/Desktop/Machine Learning/models/model_HDFC.h5')
model_bhartartl = tf.keras.models.load_model('/Users/abhyudoysmac/Desktop/Machine Learning/models/model_BHARTIARTL.h5')
model_reliance = tf.keras.models.load_model('/Users/abhyudoysmac/Desktop/Machine Learning/models/model_RELIANCE.h5')
model_hcltech = tf.keras.models.load_model('/Users/abhyudoysmac/Desktop/Machine Learning/models/model_HCLTECH.h5')

# Load the stock CSV data
def load_stock_data(stock):
    file_name = f"data/{stock}.csv"  # Assuming the CSV files are stored in 'data' directory
    df = pd.read_csv(file_name)
    return df

# Prepare data for model prediction (assuming we're predicting VWAP using ['Open', 'High', 'Low', 'Volume'])
def preprocess_data(df):
    features = ['Open', 'High', 'Low', 'Volume']
    # Select only the relevant columns ('Open', 'High', 'Low', 'Volume')
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])
    X = pd.DataFrame(columns=features, data=X, index=df.index)

    # Reshape the data to include the time-step dimension (Shape: (batch_size, 1, 4))
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))  # Reshape to (batch_size, 1, 4)
    
    return X

# Create the Dash application
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Stock Price VWAP Prediction using NIFTY50 Stock Data"),
    # Dropdown for stock selection
    dcc.Dropdown(
        id='stock-dropdown',
        options=[
            {'label': 'TCS', 'value': 'TCS'},
            {'label': 'HDFC', 'value': 'HDFC'},
            {'label': 'BHARTIARTL', 'value': 'BHARTIARTL'},
            {'label': 'RELIANCE', 'value': 'RELIANCE'},
            {'label': 'HCLTECH', 'value': 'HCLTECH'}
        ],
        value='TCS',  # Default value
        style={'width': '50%'}
    ),
    
    # RangeSlider for date selection
    dcc.RangeSlider(
        id='date-slider',
        min=0,
        max=1000,  # You may want to change this based on the size of your dataset
        step=1,
        marks={i: f'Day {i}' for i in range(0, 1001, 100)},  # Adjust marks based on dataset
        value=[0, 1000]  # Default range
    ),
    
    # Graph to show stock data and predictions
    dcc.Graph(id='stock-graph'),
    
    # Optional: Raw data display
    html.Div(id='raw-data-table')
])

@app.callback(
    [Output('stock-graph', 'figure'),
     Output('raw-data-table', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('date-slider', 'value')]
)
def update_graph(selected_stock, selected_dates):
    # Load the data for the selected stock
    df = load_stock_data(selected_stock)

    # Select the range of data based on the slider input
    start_date, end_date = selected_dates
    df = df.iloc[start_date:end_date]

    # Ensure Date is in datetime format for plotting
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert Date to datetime

    # Preprocess data (using Open, High, Low, Volume)
    X = preprocess_data(df)

    # Predict using the corresponding model
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

    # Predict the VWAP values
    y_pred = model.predict(X)

    # Prepare the graph data
    trace1 = go.Scatter(
        x=df['Date'],  # Actual dates for x-axis
        y=df['VWAP'],  # Actual VWAP values
        mode='lines',
        name='True VWAP'
    )
    trace2 = go.Scatter(
        x=df['Date'],
        y=y_pred.flatten(),  # Flatten the prediction array to match the shape
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

    # Optionally display raw data as a table
    features = ['Open', 'High', 'Low', 'Volume', 'VWAP']
    raw_data_table = html.Div(
    children=html.Table([
        html.Tr([html.Th(col) for col in features])  # Column headers
    ] + [
        html.Tr([html.Td(df.iloc[i][col]) for col in features])  # Data rows
        for i in range(len(df))
    ]),
    style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'text-align': 'center'}
)

    return figure, raw_data_table


if __name__ == '__main__':
    app.run_server(debug=True)
