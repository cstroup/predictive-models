import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from datetime import datetime, timedelta
from prophet import Prophet

ticker = "WMT"
volumn_avg = 100  # Increase smoothing on volume (100-day avg)
sma_value = 400   # Increase smoothing on SMA (400-day avg)
predictor_col = 'yhat'  # trend; yhat

# Set Plotly renderer for VS Code
pio.renderers.default = "browser"

# Load Stock Data
start_d = "2018-01-01"
yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
data = yf.download(ticker, start=start_d, end=yesterday)

# Keep relevant columns
df = data[['Close', 'Volume']].reset_index()
df.columns = ['ds', 'y', 'Volume']  # Prophet needs 'ds' for date and 'y' for value

# Compute N-day rolling average of Volume
df['Volume_Avg'] = df['Volume'].rolling(window=volumn_avg).mean()

# Compute N-day moving average of Closing Price
df['SMA'] = df['y'].rolling(sma_value).mean()

# Handle missing values
df.bfill(inplace=True)  

# Save dataset to CSV
df.to_csv(f'{ticker}_{start_d}_{str(yesterday)}_data.csv', index=False)

# --- Prophet Model: Using BOTH SMA and Volume_Avg ---
prophet_model = Prophet(changepoint_prior_scale=0.05)  # Reduce trend sensitivity
prophet_model.add_regressor('SMA')          # Adding 400-day Moving Average
prophet_model.add_regressor('Volume_Avg')   # Adding 100-day Volume Average

# Fit the model
prophet_model.fit(df[['ds', 'y', 'SMA', 'Volume_Avg']])

# Create Future DataFrame for Predictions (365 Days Ahead)
future = prophet_model.make_future_dataframe(periods=365)

# Merge additional features into future dataframe
future = future.merge(df[['ds', 'SMA', 'Volume_Avg']], on='ds', how='left')

# Fix NaN Values
future.bfill(inplace=True)
future.ffill(inplace=True)

# Make predictions
forecast = prophet_model.predict(future)

# Apply Moving Average Smoothing to Predictions (10-day avg)
forecast['yhat_smooth'] = forecast['yhat'].rolling(window=10).mean()

# Extract historical and future data separately
historical_data = forecast[forecast['ds'] <= df['ds'].max()]
future_data = forecast[forecast['ds'] > df['ds'].max()]

# Save future predictions to CSV file
future_data.to_csv(f'{ticker}_future_combined_model_data.csv', index=False)

# Get date range for last 1 year and next 1 year
one_year_ago = df['ds'].max() - timedelta(days=365)
one_year_future = df['ds'].max() + timedelta(days=365)

# --- Plotly Interactive Visualization ---
fig = go.Figure()

# Historical Data (Actual)
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual', line=dict(color='white')))

# Prophet Predictions (Past)
fig.add_trace(go.Scatter(x=historical_data['ds'], y=historical_data['yhat'], mode='lines',
                         name='Prophet Predicted (Past)', line=dict(dash='dot', color='blue')))

# Prophet Future Predictions (Smoothed)
fig.add_trace(go.Scatter(x=future_data['ds'], y=future_data['yhat_smooth'], mode='lines',
                         name='Prophet Predicted (Future, Smoothed)', line=dict(dash='dot', color='green')))

# Layout Customization - Restrict X-Axis to Last 1 Year + Next 1 Year
fig.update_layout(title=f"{ticker} Forecast with Prophet (SMA + Volume_Avg, Smoothed)",
                  xaxis_title="Date", yaxis_title="Stock Price",
                  xaxis=dict(range=[one_year_ago, one_year_future]),
                  template="plotly_dark", hovermode="x")

# Show Interactive Plot
fig.show()
