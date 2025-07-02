import pandas as pd
from prophet import Prophet

def preprocess_data(df, store_id, dept_id):
    
    # Filter by Store and Department
    filtered_df = df[(df['Store'] == store_id) & (df['Dept'] == dept_id)]
    
    # Group by Date and sum sales (in case multiple rows exist per date)
    grouped = filtered_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    
    # Rename columns as expected by Prophet
    grouped.columns = ['ds', 'y']
    
    # Convert date to datetime
    grouped['ds'] = pd.to_datetime(grouped['ds'])

    return grouped


from prophet import Prophet

def train_forecast_model(df, periods):
    model = Prophet(weekly_seasonality=True)  # ✅ Enable weekly seasonality
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    return forecast, model
