import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
from forecast_utils import preprocess_data, train_forecast_model

# Load training data
df_all = pd.read_csv('data/train.csv')

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Retail Sales Forecasting"

# Dropdown options
store_options = [{'label': f'Store {s}', 'value': s} for s in sorted(df_all['Store'].unique())]
dept_options = [{'label': f'Dept {d}', 'value': d} for d in sorted(df_all['Dept'].unique())]

# Layout
app.layout = html.Div([
    html.H1("🛍️ Retail Sales Forecast Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Store:"),
        dcc.Dropdown(id='store-dropdown', options=store_options, value=1),

        html.Br(),

        html.Label("Select Department:"),
        dcc.Dropdown(id='dept-dropdown', options=dept_options, value=1),

        html.Br(),

        html.Label("Forecast Weeks Ahead:"),
        dcc.Slider(
            id='forecast-weeks',
            min=4, max=52, step=4, value=12,
            marks={i: f'{i}w' for i in range(4, 53, 4)}
        ),
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Br(),

    dcc.Graph(id='forecast-graph')
])

# Callback to update forecast
@app.callback(
    Output('forecast-graph', 'figure'),
    Input('store-dropdown', 'value'),
    Input('dept-dropdown', 'value'),
    Input('forecast-weeks', 'value')
)
def update_forecast(store_id, dept_id, weeks):
    df = preprocess_data(df_all, store_id, dept_id)
    forecast, model = train_forecast_model(df, periods=weeks)

    # Plot forecast and actual sales
    fig = px.line(forecast, x='ds', y='yhat', title=f"Forecast: Store {store_id}, Dept {dept_id}")
    fig.add_scatter(x=df['ds'], y=df['y'], mode='lines+markers', name='Actual Sales')
    fig.update_layout(xaxis_title="Date", yaxis_title="Weekly Sales")
    return fig

# Run app
if __name__ == '__main__':
    app.run(debug=True)          

