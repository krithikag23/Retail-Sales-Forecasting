import pandas as pd
from prophet import Prophet

def preprocess_data(df, store_id, dept_id):
    """
    Filters the dataset for a specific store and department,
    groups weekly sales by date, and formats it for Prophet.
    """
    # Filter by Store and Department
    filtered_df = df[(df['Store'] == store_id) & (df['Dept'] == dept_id)]
    
    # Group by Date and sum sales (in case multiple rows exist per date)
    grouped = filtered_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    
    # Rename columns as expected by Prophet
    grouped.columns = ['ds', 'y']
    
    # Convert date to datetime
    grouped['ds'] = pd.to_datetime(grouped['ds'])

    return grouped


def train_forecast_model(df, periods=12):
    """
    Trains a Prophet model on the given dataframe and returns the forecast.
    """
    model = Prophet()

    # Fit the model
    model.fit(df)

    # Create future dataframe (weekly frequency)
    future = model.make_future_dataframe(periods=periods, freq='W')

    # Forecast
    forecast = model.predict(future)

    return forecast, model
