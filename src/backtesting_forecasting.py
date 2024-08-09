
import numpy as np
import pandas as pd
import warnings 

def hist_forecast(df, target, features, split_time, horizon, y_lags, model, retrain=True):
    """
    Generate historical forecasts for a given dataset using a specified model.
    
    Parameters:
    - df (pd.DataFrame): The dataset containing features and target.
    - target (str): The target column to forecast.
    - features (list of str): List of feature columns to use for forecasting.
    - split_time (str or pd.Timestamp): The point in time to split the data into training and validation.
    - horizon (int): The number of steps ahead to forecast.
    - y_lags (int): Number of lag periods to include as features.
    - model (object): The model used for forecasting.
    - retrain (bool): Whether to retrain the model at each prediction point.
    
    Returns:
    - list: The forecasted values.
    """
    # Ensure lagged features are included
    lagged_features = [f"{target}_lag_{j}" for j in range(y_lags, 0, -1) if f"{target}_lag_{j}" not in features]
    features = lagged_features + features

    # Split the data into training and validation sets
    val_df = df[features][split_time:]
    train_y = df[target][:split_time] 
    train_df = df[features][:split_time]

    if retrain:
        model.fit(train_df, train_y) 

    # Start the forecast with the last y_lags values from training set
    forecast = [train_y.iloc[-i] for i in range(y_lags, 0, -1)]

    for i in range(horizon):
        X = np.array(forecast[-y_lags:] + list(val_df[features[y_lags:]].iloc[i])).reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = model.predict(X)[0]
        forecast.append(y_pred)

    return forecast[y_lags:]

def backtesting(df, target, features, test_split_time, horizon, y_lags, model, retrain=True):
    """
    Perform backtesting by generating forecasts for multiple periods in the test set.
    
    Parameters:
    - df (pd.DataFrame): The dataset containing features and target.
    - target (str): The target column to forecast.
    - features (list of str): List of feature columns to use for forecasting.
    - test_split_time (str or pd.Timestamp): The start of the test period.
    - horizon (int): The number of steps ahead to forecast at each period.
    - y_lags (int): Number of lag periods to include as features.
    - model (object): The model used for forecasting.
    - retrain (bool): Whether to retrain the model at each prediction point.
    
    Returns:
    - list: The forecasted values for the entire test period.
    """
    pred_times = [
        t for i, t in enumerate(pd.date_range(start=test_split_time, end=df.index[-1], freq=df.index.inferred_freq))
        if i % horizon == 0
    ]
    forecast = []
    for t in pred_times:
        forecast_i = hist_forecast(df, target, features, t, horizon, y_lags, model, retrain)
        forecast.append(forecast_i)

    return [pred for sublist in forecast for pred in sublist]

def predict(df, target, features, horizon, y_lags, model, retrain=True):
    """
    Generate forecasts for a given dataset using a specified model.

    Parameters:
    - df (pd.DataFrame): The dataset containing features and target.
    - target (str): The target column to forecast.
    - features (list of str): List of feature columns to use for forecasting.
    - horizon (int): The number of steps ahead to forecast.
    - y_lags (int): Number of lag periods to include as features.
    - model (object): The model used for forecasting.
    - retrain (bool): Whether to retrain the model before forecasting.

    Returns:
    - pd.Series: The forecasted values with corresponding dates as the index.
    """
    # Extract base feature names (without lag suffixes)
    base_features = [f[:-6] for f in features]

    # Prepare the validation DataFrame with future dates for the forecast horizon
    val_df = pd.DataFrame(
        np.array(df[base_features].iloc[-horizon:]), 
        columns=features, 
        index=pd.date_range(start=df.index[-1], periods=horizon + 1, freq=df.index.inferred_freq)[1:]
    )

    # Ensure lagged features are included
    lagged_features = [f"{target}_lag_{j}" for j in range(y_lags, 0, -1) if f"{target}_lag_{j}" not in features]
    features = lagged_features + features

    # Prepare the training set
    train_y = df[target]
    train_df = df[features]

    # Retrain the model if required
    if retrain:
        model.fit(train_df, train_y)

    # Initialize the forecast with the last y_lags values from the training target
    forecast = [train_y.iloc[-i] for i in range(y_lags, 0, -1)]

    # Generate forecasts for each step in the horizon
    for i in range(horizon):
        X = np.array(forecast[-y_lags:] + list(val_df.iloc[i])).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = model.predict(X)[0]
        forecast.append(y_pred)

    # Convert the forecast list to a pandas Series
    forecast = pd.Series(forecast[y_lags:], name="Forecast", index=val_df.index)
    return forecast