import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.tools as tls
from forecast_utils import preprocess_data, train_forecast_model
from dash import Input, Output
import plotly.express as px

# Load training data
df_all = pd.read_csv('data/train.csv')

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Retail Sales Forecasting using Prophet"

# Dropdown options
store_options = [{'label': f'Store {s}', 'value': s} for s in sorted(df_all['Store'].unique())]
dept_options = [{'label': f'Dept {d}', 'value': d} for d in sorted(df_all['Dept'].unique())]

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

        html.Label("Forecast Weeks Ahead:"),
        dcc.Slider(
            id='forecast-weeks',
            min=4, max=52, step=4, value=12,
            marks={i: f'{i}w' for i in range(4, 53, 4)}
        ),
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Br(),

    dcc.Graph(id='forecast-graph'),
    html.Br(),
    dcc.Graph(id='trend-graph'),
    html.Br(),
    dcc.Graph(id='seasonality-graph')
])



@app.callback(
    Output('forecast-graph', 'figure'),
    Output('trend-graph', 'figure'),
    Output('seasonality-graph', 'figure'),
    Input('store-dropdown', 'value'),
    Input('dept-dropdown', 'value'),
    Input('forecast-weeks', 'value')
)
def update_all_graphs(store_id, dept_id, weeks):
    df = preprocess_data(df_all, store_id, dept_id)
    forecast, model = train_forecast_model(df, periods=weeks)

    # 📈 Forecast Graph
    fig_forecast = px.line(forecast, x='ds', y='yhat', title=f"Forecast: Store {store_id}, Dept {dept_id}")
    fig_forecast.add_scatter(x=df['ds'], y=df['y'], mode='lines+markers', name='Actual Sales')
    fig_forecast.update_layout(xaxis_title="Date", yaxis_title="Weekly Sales")

    # 🔍 Trend Graph
    trend_fig = px.line(forecast, x='ds', y='trend', title="Trend Component")
    trend_fig.update_layout(xaxis_title="Date", yaxis_title="Trend")

    # 🔁 Weekly Seasonality
    weekly = forecast[['ds', 'weekly']].dropna().copy()
    weekly['day'] = weekly['ds'].dt.day_name()
    weekly_avg = weekly.groupby('day')['weekly'].mean().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_avg['day'] = pd.Categorical(weekly_avg['day'], categories=day_order, ordered=True)
    weekly_avg = weekly_avg.sort_values('day')

    seasonality_fig = px.bar(weekly_avg, x='day', y='weekly', title='Weekly Seasonality',
                              labels={'day': 'Day of Week', 'weekly': 'Seasonal Effect'})

    return fig_forecast, trend_fig, seasonality_fig


# Run app
if __name__ == '__main__':
    app.run(debug=True)
