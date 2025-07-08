import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
from forecast_utils import preprocess_data, train_forecast_model

# Load training data
df_all = pd.read_csv('data/train.csv')
df_all['Date'] = pd.to_datetime(df_all['Date'])  # Ensure datetime format

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Retail Sales Forecasting using Prophet"

# Dropdown options
store_options = [{'label': f'Store {s}', 'value': s} for s in sorted(df_all['Store'].unique())]
dept_options = [{'label': f'Dept {d}', 'value': d} for d in sorted(df_all['Dept'].unique())]
freq_options = [
    {'label': 'Daily', 'value': 'D'},
    {'label': 'Weekly', 'value': 'W'},
    {'label': 'Monthly', 'value': 'M'},
    {'label': 'Yearly', 'value': 'Y'},
]
model_optioms= {' '}
# Layout
app.layout = html.Div([
    html.H1("🛍️ Retail Sales Forecast Dashboard (using Prophet)", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Store:"),
        dcc.Dropdown(id='store-dropdown', options=store_options, value=1),

        html.Br(),

        html.Label("Select Department:"),
        dcc.Dropdown(id='dept-dropdown', options=dept_options, value=1),

        html.Br(),

        html.Label("Select Forecast Frequency:"),
        dcc.Dropdown(id='freq-dropdown', options=freq_options, value='W'),

        html.Br(),

        html.Label("Forecast Horizon:"),
        dcc.Slider(id='forecast-slider', min=4, max=52, step=1, value=12),

        html.Div(id='slider-label', style={"marginTop": "10px", "fontWeight": "bold"})
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Br(),
    dcc.Graph(id='forecast-graph'),
    html.Br(),
    dcc.Graph(id='trend-graph'),
    html.Br(),
    dcc.Graph(id='seasonality-graph')
])

# Dynamically update slider range based on frequency
@app.callback(
    Output('forecast-slider', 'min'),
    Output('forecast-slider', 'max'),
    Output('forecast-slider', 'step'),
    Output('forecast-slider', 'value'),
    Output('forecast-slider', 'marks'),
    Output('slider-label', 'children'),
    Input('freq-dropdown', 'value')
)
def update_slider(freq):
    if freq == 'D':
        return 7, 90, 1, 30, {i: f'{i}d' for i in range(7, 91, 7)}, "Forecast Days Ahead:"
    elif freq == 'W':
        return 4, 52, 1, 12, {i: f'{i}w' for i in range(4, 53, 4)}, "Forecast Weeks Ahead:"
    elif freq == 'M':
        return 1, 24, 1, 6, {i: f'{i}m' for i in range(1, 25)}, "Forecast Months Ahead:"
    elif freq == 'Y':
        return 1, 5, 1, 2, {i: f'{i}y' for i in range(1, 6)}, "Forecast Years Ahead:"
    return 4, 52, 1, 12, {}, ""

# Update all graphs based on input
@app.callback(
    Output('forecast-graph', 'figure'),
    Output('trend-graph', 'figure'),
    Output('seasonality-graph', 'figure'),
    Input('store-dropdown', 'value'),
    Input('dept-dropdown', 'value'),
    Input('forecast-slider', 'value'),
    Input('freq-dropdown', 'value')
)
def update_all_graphs(store_id, dept_id, periods, freq):
    df = preprocess_data(df_all, store_id, dept_id)
    forecast, model = train_forecast_model(df, periods=periods, freq=freq)

    forecast['ds'] = pd.to_datetime(forecast['ds'])

    # Forecast Graph
    fig_forecast = px.line(forecast, x='ds', y='yhat', title=f"Forecast: Store {store_id}, Dept {dept_id}")
    fig_forecast.add_scatter(x=df['ds'], y=df['y'], mode='lines+markers', name='Actual Sales')
    fig_forecast.update_layout(xaxis_title="Date", yaxis_title="Sales")

    # Trend Component
    trend_fig = px.line(forecast, x='ds', y='trend', title="Trend Component")
    trend_fig.update_layout(xaxis_title="Date", yaxis_title="Trend")

    # Weekly Seasonality
    if 'weekly' in forecast.columns:
        weekly = forecast[['ds', 'weekly']].dropna().copy()
        weekly['day'] = weekly['ds'].dt.day_name()
        weekly_avg = weekly.groupby('day')['weekly'].mean().reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_avg['day'] = pd.Categorical(weekly_avg['day'], categories=day_order, ordered=True)
        weekly_avg = weekly_avg.sort_values('day')

        seasonality_fig = px.bar(weekly_avg, x='day', y='weekly', title='Weekly Seasonality',
                                 labels={'day': 'Day of Week', 'weekly': 'Seasonal Effect'})
    else:
        seasonality_fig = px.bar(title='Seasonality not available for this frequency.')

    return fig_forecast, trend_fig, seasonality_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
