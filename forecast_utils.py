from prophet import Prophet

def preprocess_data(df_all, store_id, dept_id):
    df = df_all[(df_all['Store'] == store_id) & (df_all['Dept'] == dept_id)].copy()
    df.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'}, inplace=True)
    df = df[['ds', 'y']].sort_values('ds')
    return df

def train_forecast_model(df, periods, freq='W'):
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    return forecast, model
