# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:15:55 2024

@author: Akshay Singh
"""

# prepared by Akshay Singh

import yfinance as yf
import pandas_datareader.data as pdr
import pandas as pd
import numpy as np
import statsmodels.api as sm
from darts import TimeSeries
from darts.models import LinearRegressionModel, AutoARIMA, Prophet, XGBModel
import matplotlib.pyplot as plt


start_date = '2001-01-01'
end_date = '2023-01-01'

# Consumer Price Index (CPI)
cpi = pdr.DataReader('CPIAUCSL', 'fred', start=start_date, end=end_date)
cpi.rename(columns={'CPIAUCSL': 'CPI'}, inplace=True)
cpi = cpi.resample('ME').last() #'ME' - month ending (month frequency)

# Crude Oil Prices
crude_oil = yf.download('CL=F', start=start_date, end=end_date)['Adj Close']
crude_oil = crude_oil.rename('Crude Oil Price')
crude_oil = crude_oil.resample('ME').last() #'ME' - month ending (month frequency)
crude_oil = crude_oil.to_frame(name="Crude Oil Price")

# Unemployment Rate
unemployment = pdr.DataReader('UNRATE', 'fred', start=start_date, end=end_date)
unemployment.rename(columns={'UNRATE': 'Unemployment Rate'}, inplace=True)
unemployment = unemployment.resample('ME').last() #'ME' - month ending (month frequency)


data = pd.concat([cpi, crude_oil, unemployment], axis=1, join='inner')

# lag for CPI
data['CPI Lag 1'] = data['CPI'].shift(1)
data['CPI Lag 2'] = data['CPI'].shift(2)
data['CPI Lag 3'] = data['CPI'].shift(3)

# lag for Unemployment
data['Unemployment Lag 1'] = data['Unemployment Rate'].shift(1)
data['Unemployment Lag 2'] = data['Unemployment Rate'].shift(2)
data['Unemployment Lag 3'] = data['Unemployment Rate'].shift(3)

# Dropping missing values
data.dropna(inplace=True)


# X and Y variable 
y = data['Crude Oil Price']
x_cpi = data[['CPI', 'CPI Lag 1', 'CPI Lag 2', 'CPI Lag 3']]
x_cpi = sm.add_constant(x_cpi)

x_unemployment = data[['Unemployment Rate', 'Unemployment Lag 1', 'Unemployment Lag 2', 'Unemployment Lag 3']]
x_unemployment = sm.add_constant(x_unemployment)


# OLS for CPI
model_cpi = sm.OLS(y, x_cpi).fit()
print('OLS Regression Results for CPI:')
print(model_cpi.summary())

# OLS for unemployment 
model_unemployment = sm.OLS(y, x_unemployment).fit()
print('OLS Regression Results for Unemployment Rate:')
print(model_unemployment.summary())

# We are trying to show how the past values of CPI influences the Crude oil price and similarly how Unemployment affects the Crude Oil prices 
# CPI in the OLS regression model suggests that CPI with P <0.05 (confidence level) is more than P value of 0.026 which reflects the relationship is statistically significant 
# The CPI lagged and current CPI shows the relation is significant meaning that an increase in current CPI is associated with Crude Oil Price Increase
#%% - Time Series Modeling and Forecasting

# Time series for CPI
series_cpi = TimeSeries.from_dataframe(cpi, value_cols='CPI', freq='ME')

# Linear Regression Model for CPI
train_cpi_lr, val_cpi_lr = series_cpi.split_before(0.8)
lr_model = LinearRegressionModel(lags=12)  # 12 months of lag to predict
lr_model.fit(train_cpi_lr)
forecast_cpi = lr_model.predict(len(val_cpi_lr))

plt.figure(figsize=(10, 6))
train_cpi_lr.plot(label='Train')
val_cpi_lr.plot(label='Actual (Validation)')
forecast_cpi.plot(label='Forecast (Linear Regression)')
plt.title('Actual vs Predicted - Linear Regression - CPI')
plt.legend()
plt.show()

# Future Linear Regression Forecast for CPI
lr_model.fit(series_cpi)
forecast_horizon_future = 24
lr_future_forecast_cpi = lr_model.predict(forecast_horizon_future)

plt.figure(figsize=(10, 6))
series_cpi.plot(label='Full Sample')
lr_future_forecast_cpi.plot(label='Linear Regression Forecast', lw=2)
plt.title('Future Forecast - Linear Regression (Next 24 Months) - CPI')
plt.legend()
plt.show()

## Future LR Forecast model performed better as it shows a smooth, linear projection of CPI into the future, implying that the model expects the CPI to continue increasing at a steady rate.
## LR model Indicates that the linear regression model may not perfectly capture the nuances and sudden changes in the CPI, as evidenced by the divergence between the forecasted and actual validation data. 

# ARIMA and Prophet Models for CPI
arima_model = AutoARIMA()
prophet_model = Prophet()

train_cpi, val_cpi = series_cpi.split_before(0.9)
arima_model.fit(train_cpi)
prophet_model.fit(train_cpi)

forecast_horizon = len(val_cpi)
arima_forecast_cpi = arima_model.predict(forecast_horizon)
prophet_forecast_cpi = prophet_model.predict(forecast_horizon)

plt.figure(figsize=(10, 6))
train_cpi.plot(label='Train')
val_cpi.plot(label='Actual', lw=2)
arima_forecast_cpi.plot(label='ARIMA Forecast', lw=2)
prophet_forecast_cpi.plot(label='PROPHET Forecast', lw=2)
plt.title('Model Comparison - CPI')
plt.legend()
plt.show()

# Future ARIMA and Prophet Forecast for CPI
arima_model.fit(series_cpi)
prophet_model.fit(series_cpi)

forecast_horizon_future = 24
arima_future_forecast_cpi = arima_model.predict(forecast_horizon_future)
prophet_future_forecast_cpi = prophet_model.predict(forecast_horizon_future)

plt.figure(figsize=(10, 6))
series_cpi.plot(label='Full Sample')
arima_future_forecast_cpi.plot(label='ARIMA Forecast', lw=2)
prophet_future_forecast_cpi.plot(label='PROPHET Forecast', lw=2)
plt.title('Future Forecast (Next 24 Months) - CPI')
plt.legend()
plt.show()

## Prophet Model is much better in comapring the above two models as it accounts for the seasonal trend as well
## As seen in the first graph of Model comparison the Prophet and ARIMA model follow upward trend whereas in the Future forecast of next 24 month CPI prophet continues to account for seasonal trend whereas ARIMA deviates to upward trend. 

# XGBoost Model for CPI
xgb_model = XGBModel(lags=12)
xgb_model.fit(train_cpi)
forecast_horizon = len(val_cpi)
num_simulations = 20

# XGBoost Model for CPI
xgb_model = XGBModel(lags=12)
xgb_model.fit(train_cpi)
forecast_horizon = len(val_cpi)
num_simulations = 20

xgb_simulations_cpi = [xgb_model.predict(forecast_horizon, num_samples=1).pd_dataframe().values.flatten() for _ in range(num_simulations)]
xgb_mean_forecast_cpi = np.mean(xgb_simulations_cpi, axis=0)

plt.figure(figsize=(10, 6))
train_cpi.plot(label='Train')
val_cpi.plot(label='Actual (Validation)', lw=2)
plt.plot(val_cpi.time_index, xgb_mean_forecast_cpi, label='XGBoost Mean Forecast', color='red')
plt.title('Monte Carlo Simulation: XGBoost Forecast for CPI')
plt.legend()
plt.show()

# Future XGBoost Forecast for CPI
# Refit models on the full dataset
xgb_model.fit(series_cpi)
forecast_horizon_future = 24

# XGBoost future forecast
xgb_future_forecast_cpi = (xgb_model.predict(forecast_horizon_future).pd_dataframe().values.flatten())

# Create a date range for the future forecast
future_dates = pd.date_range(start=series_cpi.time_index[-1], periods=forecast_horizon_future + 1, freq='ME')[1:]

# Plot future forecasts
plt.figure(figsize=(10, 6))
series_cpi.plot(label='Full Sample')

# Plot XGBoost forecast
plt.plot(future_dates, xgb_future_forecast_cpi, label='XGBoost Future Forecast', color='red')
plt.title('Future Forecast: XGBoost (Next 24 Months) - CPI')
plt.legend()
plt.show()

## Both the models provide the insights on trends of CPI
## I think monte carlo is better comparing XGBoost Future Forecast as it accounts for uncertainity providing a more robust forecast while XGBoost shows the rising trend of CPI over extrapolation of next 24 months
## Also mean forecast from multiple XGBoost simulations (red line) aligns closely with the actual validation data (blue line), indicating that the model effectively captures the historical CPI trend. 


## Time series for Unemployment Rate
series_unemployment = TimeSeries.from_dataframe(unemployment, value_cols='Unemployment Rate', freq='ME')

# Linear Regression Model for Unemployment Rate
train_unemployment_lr, val_unemployment_lr = series_unemployment.split_before(0.8)
lr_model.fit(train_unemployment_lr)
forecast_unemployment = lr_model.predict(len(val_unemployment_lr))

plt.figure(figsize=(10, 6))
train_unemployment_lr.plot(label='Train')
val_unemployment_lr.plot(label='Actual (Validation)')
forecast_unemployment.plot(label='Forecast (Linear Regression)')
plt.title('Actual vs Predicted - Linear Regression - Unemployment Rate')
plt.legend()
plt.show()

# Future Linear Regression Forecast for Unemployment Rate
lr_model.fit(series_unemployment)
forecast_horizon_future = 24
lr_future_forecast_unemployment = lr_model.predict(forecast_horizon_future)

plt.figure(figsize=(10, 6))
series_unemployment.plot(label='Full Sample')
lr_future_forecast_unemployment.plot(label='Linear Regression Forecast', lw=2)
plt.title('Future Forecast - Linear Regression (Next 24 Months) - Unemployment Rate')
plt.legend()
plt.show()


## The actual vs predicted - Linear regression model validates the past data and predicts actual during validation period. helping to assess model reliability
## The Future Forecast - Linear regression (24 Months) forecast data for next 24 months. While the goal here is to predict the data for future, Future forecast - LR performs better is showcasing the trend
## both the model show upward trend but future forecast performs better in extrapolating for addtional two years as per the sample data.

# ARIMA and Prophet Models for Unemployment Rate
train_unemployment, val_unemployment = series_unemployment.split_before(0.9)
arima_model.fit(train_unemployment)
prophet_model.fit(train_unemployment)

forecast_horizon = len(val_unemployment)
arima_forecast_unemployment = arima_model.predict(forecast_horizon)
prophet_forecast_unemployment = prophet_model.predict(forecast_horizon)

plt.figure(figsize=(10, 6))
train_unemployment.plot(label='Train')
val_unemployment.plot(label='Actual', lw=2)
arima_forecast_unemployment.plot(label='ARIMA Forecast', lw=2)
prophet_forecast_unemployment.plot(label='PROPHET Forecast', lw=2)
plt.title('Model Comparison - Unemployment Rate')
plt.legend()
plt.show()

# Future ARIMA and Prophet Forecast for Unemployment Rate
arima_model.fit(series_unemployment)
prophet_model.fit(series_unemployment)

forecast_horizon_future = 24
arima_future_forecast_unemployment = arima_model.predict(forecast_horizon_future)
prophet_future_forecast_unemployment = prophet_model.predict(forecast_horizon_future)

plt.figure(figsize=(10, 6))
series_unemployment.plot(label='Full Sample')
arima_future_forecast_unemployment.plot(label='ARIMA Forecast', lw=2)
prophet_future_forecast_unemployment.plot(label='PROPHET Forecast', lw=2)
plt.title('Future Forecast (Next 24 Months) - Unemployment Rate')
plt.legend()
plt.show()

## Both the models shows trend for unemployment based on the historical data, however the PROPHET forecast in the both the graph provides an accurate forecast accounting for the seasonality and the trend shifts 
## the Prophet forecast line in both the model graph remains unchanged showing the accurate display while the ARIMA model deviates in the forecasting showing flat line in the trend compared to dip in the first graph 

# XGBoost Model for Unemployment Rate
xgb_model.fit(train_unemployment)
forecast_horizon = len(val_unemployment)
num_simulations = 20

xgb_simulations_unemployment = [xgb_model.predict(forecast_horizon, num_samples=1).pd_dataframe().values.flatten() for _ in range(num_simulations)]
xgb_mean_forecast_unemployment = np.mean(xgb_simulations_unemployment, axis=0)

plt.figure(figsize=(10, 6))
train_unemployment.plot(label='Train')
val_unemployment.plot(label='Actual (Validation)', lw=2)
plt.plot(val_unemployment.time_index, xgb_mean_forecast_unemployment, label='XGBoost Mean Forecast', color='red')
plt.title('Monte Carlo Simulation: XGBoost Forecast for Unemployment Rate')
plt.legend()
plt.show()

# Future XGBoost Forecast for Unemployment Rate

xgb_model.fit(series_unemployment)
forecast_horizon_future = 24

# XGBoost future forecast
xgb_future_forecast_unemployment = (xgb_model.predict(forecast_horizon_future).pd_dataframe().values.flatten())


future_dates = pd.date_range(start=series_unemployment.time_index[-1], periods=forecast_horizon_future + 1, freq='ME')[1:]

# Plot of Future forecast
plt.figure(figsize=(10, 6))
series_unemployment.plot(label='Full Sample')

# Plot
plt.plot(future_dates, xgb_future_forecast_unemployment, label='XGBoost Future Forecast', color='red')
plt.title('Future Forecast: XGBoost (Next 24 Months) - Unemployment Rate')
plt.legend()
plt.show()

## Both the models provide the insights on trends of unemployment, 
## I think monte carlo is better comparing XGBoost Future Forecast as it accounts for uncertainity providing a more robust forecast while XGBoost shows the declining trend of Unemployment over extrapolation of next 24 months
## Also mean forecast from multiple XGBoost simulations (red line) aligns closely with the actual validation data (blue line), indicating that the model effectively captures the historical Unemployment trend. 
