import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io
import base64

def dataset(symbol):
    data = yf.download(symbol, period="1y")["Close"]
    all_days = pd.date_range(start=data.index.min(), end=data.index.max(), freq="B")
    data = data.reindex(all_days)
    data = data.ffill()
    data = data.reset_index()
    long_df = data.melt(id_vars=["index"], 
                    var_name="unique_id", 
                    value_name="y")
    long_df = long_df.rename(columns={"index": "ds"})
    for lag in [1, 7]:
        long_df[f"y_lag_{lag}"] = long_df.groupby("unique_id")["y"].shift(lag)
    long_df["rolling_mean_7"] = long_df.groupby("unique_id")["y"].shift(1).rolling(7).mean()
    long_df["day_of_week"] = long_df["ds"].dt.dayofweek 
    long_df = long_df.dropna()
    long_df = long_df.reset_index(drop=True)
    return long_df



#  Prepare Data for one stock 
def prepare_stock_data(long_df, symbol):
    """
    Filter long_df for a single stock symbol
    and return X, y sorted by date.
    """
    df = long_df[long_df['unique_id'] == symbol].sort_values('ds')
    feature_cols = ['y_lag_1', 'y_lag_7', 'rolling_mean_7', 'day_of_week']
    X = df[feature_cols]
    y = df['y']
    dates = df['ds']
    return X, y, dates




# ---------------- Forecast next 7 days ----------------
def recursive_forecast(X, y, model, n_days=7):
    """
    Recursive forecast: predict day t+1, append, predict t+2, ...
    """
    y_ext = y.copy().tolist()
    X_ext = X.copy()
    preds = []

    for i in range(n_days):
        x_last = X_ext.iloc[[-1]].copy()

        # Predict next day
        y_hat = model.predict(x_last)[0]
        preds.append(y_hat)

        # Create next day's features
        next_lag_1 = y_ext[-1]
        next_lag_7 = y_ext[-7] if len(y_ext) >= 7 else y_ext[0]
        next_rollmean_7 = np.mean(y_ext[-7:]) if len(y_ext) >= 7 else np.mean(y_ext)
        next_dow = (X_ext['day_of_week'].iloc[-1] + 1) % 5  # simple business day increment

        next_row = pd.DataFrame({
            'y_lag_1': [next_lag_1],
            'y_lag_7': [next_lag_7],
            'rolling_mean_7': [next_rollmean_7],
            'day_of_week': [next_dow]
        })

        X_ext = pd.concat([X_ext, next_row], ignore_index=True)
        y_ext.append(y_hat)

    return preds
    


# ---------------- Evaluate recent MAE/RMSE ----------------
def evaluate_recent(X, y, model, val_days=30):
    X_train, y_train = X[:-val_days], y[:-val_days]
    X_val, y_val = X[-val_days:], y[-val_days:]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    # rmse = mean_squared_error(y_val, y_pred, squared=False)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return mae, rmse


def plot_forecast(dates, y, forecast, symbol):
    plt.figure(figsize=(10,5))
    plt.plot(dates[-90:], y[-90:], label="Actual (last 90 days)")
    future_dates = pd.bdate_range(dates.iloc[-1]+pd.Timedelta(1,'D'), periods=len(forecast))
    plt.plot(future_dates, forecast, linestyle='--', color='red', label="Forecast (next 7 days)")
    plt.title(f"{symbol} 7-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    # plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img_base64


# ---------------- Trend direction ----------------
def classify_trend(y_last7, forecast):
    base = np.mean(y_last7)
    change = (np.mean(forecast) - base)/base
    if change > 0.005:
        return "Expected Uptrend"
    elif change < -0.005:
        return "Expected Downtrend"
    else:
        return "Expected Sideways"
    


def run_stock_forecast(long_df, symbol, model_type='rf', n_forecast=7):
    X, y, dates = prepare_stock_data(long_df, symbol)
    model = RandomForestRegressor(n_estimators=400, random_state=42)
    
    # Evaluate recent performance
    mae, rmse = evaluate_recent(X, y, model, val_days=30)  
    
    # Fit on full data and forecast next 7 days
    model.fit(X, y)
    forecast = recursive_forecast(X, y, model, n_days=n_forecast)

    # Trend
    trend = classify_trend(y[-7:], forecast)

    # Plot
    img_base64 = plot_forecast(dates, y, forecast, symbol)

    return {
        'symbol': symbol,
        'model': model_type,
        'forecast': forecast,
        'MAE': mae,
        'RMSE': rmse,
        'trend': trend
    },img_base64