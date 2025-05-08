import pandas as pd
import numpy as np
import os
from src.ETL import *
from darts.models import LinearRegressionModel
from darts import TimeSeries

DATA_PATH = "data"
FORECAST_PATH = "forecasts"

target = "MLS Price Benchmark"

df = etl_pipeline(
    extract_data_from_CREA,
    {
        "file_path": f"{DATA_PATH}/MLS_HPI/Seasonally Adjusted (M).xlsx",
        "sheet_name": "AGGREGATE",
        "in_col": "Composite_Benchmark_SA",
        "new_names": [f"{target}"],
    },
    num_lags=13,
    lagged_cols=[f"{target} growth"],
    transform_dict={f"{target}": growth(1)},
)

df_cov = df.join(
    etl_pipeline(
        extract_data_from_StatsCanada_CSV,
        {
            "file_path": f"{DATA_PATH}/34100145.csv",
            "in_col": None,
            "out_cols": ["Mortgage rate 5y"],
        },
        transform_dict={"Mortgage rate 5y": [np.log]},
    )
)


df_cov = df_cov.join(
    etl_pipeline(
        extract_data_from_StatsCanada_CSV,
        {
            "file_path": f"{DATA_PATH}/10100116.csv",
            "in_col": "Assets, liabilities and monetary aggregates",
            "out_cols": [
                "M2++ (gross) (M2+ (gross), Canada Savings Bonds, non-money market mutual funds)",
            ],
            "new_names": ["M2++"],
        },
        transform_dict={
            "M2++": [growth(1)],
        },
    )
)

target_toronto = "Toronto MLS Price Benchmark"

df_cov = df_cov.join(
    etl_pipeline(
        extract_data_from_CREA,
        {
            "file_path": f"{DATA_PATH}/MLS_HPI/Not Seasonally Adjusted (M).xlsx",
            "sheet_name": "Greater_Toronto",  #'GREATER_TORONTO',
            "in_col": "Apartment_Benchmark",
            "new_names": [target_toronto],
        },
        num_lags=13,
        lagged_cols=[f"{target_toronto} growth"],
        transform_dict={f"{target_toronto}": [growth(1)]},
    )
)

df_cov = df_cov.join(
    etl_pipeline(
        extract_data_from_HS_json,
        {
            "file_path": f"{DATA_PATH}/rent.json",
            "cols": [
                "sold_count",
                "list_count",
                "list_active",
                "rent_count",
                "rent_listing_count",
            ],
            "new_names": [
                "Total Sold",
                "New Listings",
                "Active Listings",
                "Total Leased",
                "New Rent Listings",
            ],
        },
        transform_dict={"Active Listings": [growth(1)]},
    )
)

df_cov["SNLR"] = df_cov["Total Sold"] / df_cov["New Listings"]
df_cov = make_lags(df_cov, 12, ["SNLR"])


target_rent = "Median Rent Price"

df_cov = df_cov.join(
    etl_pipeline(
        extract_data_from_HS_json,
        {
            "file_path": f"{DATA_PATH}/rent.json",
            "cols": ["price_rent"],
            "new_names": [target_rent],
        },
        transform_dict={target_rent: growth(1)},
        num_lags=13,
        lagged_cols=[f"{target_rent} growth"],
    )
)
df_cov = df_cov.join(
    etl_pipeline(
        extract_data_from_StatsCanada_API,
        {"vector_id": 1235049756, "latest_n": 240, "new_name": "Unemployment rate"},
        transform_dict={
            "Unemployment rate": [diff(1)],
        },
    )
)

df_cov = df_cov.join(
    etl_pipeline(
        extract_data_from_StatsCanada_API,
        {"vector_id": "111955499", "latest_n": 240, "new_name": "NHPI"},
        transform_dict={"NHPI": [growth(1)]},
    )
)
df_cov["LNLR"] = df_cov["Total Leased"] / df_cov["New Rent Listings"]
df_cov = make_lags(df_cov, 12, ["LNLR"])

df_cov = df_cov.ffill().bfill()
df_cov = df_cov[sorted(df_cov.columns)]


features = ["Mortgage rate 5y log", "M2++ growth"]

LR = LinearRegressionModel(
    lags=2,
    lags_future_covariates=[-6],
    output_chunk_length=1,
)


os.makedirs("forecasts", exist_ok=True)


def generate_forecast(df, target, features, model, horizon):
    y = TimeSeries.from_dataframe(df, value_cols=f"{target} growth")
    cov = TimeSeries.from_dataframe(df, value_cols=features)
    model.fit(y, future_covariates=cov)
    forecast = model.predict(horizon, y, future_covariates=cov)

    return convert_forecast(df[[target]], forecast.pd_dataframe())


generate_forecast(df_cov, target, features, LR, 6).to_csv(
    f"{FORECAST_PATH}/forecast_hpi.csv", index=False
)

# Toronto Price Benchmark forecasting
features = ["Mortgage rate 5y log", "SNLR", "Active Listings growth"]

LR = LinearRegressionModel(
    lags=1,
    lags_future_covariates=[-3],
    output_chunk_length=1,
)

generate_forecast(df_cov, target_toronto, features, LR, 3).to_csv(
    f"{FORECAST_PATH}/forecast_hpi_toronto.csv", index=False
)

features = ["Unemployment rate diff", "NHPI growth", "LNLR"]
LR = LinearRegressionModel(
    lags=12,
    lags_future_covariates={
        "Unemployment rate diff": [-5],
        "NHPI growth": [-3, -15],
        "LNLR": [-3],
    },
    output_chunk_length=1,
)

generate_forecast(df_cov, target_rent, features, LR, 3).to_csv(
    f"{FORECAST_PATH}/forecast.csv", index=False
)

df_cov.to_csv(f"{FORECAST_PATH}/df_cov.csv")
