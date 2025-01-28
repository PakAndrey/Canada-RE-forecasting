import pandas as pd
import numpy as np

from src.ETL import *

from darts.models import LinearRegressionModel
from darts import TimeSeries, concatenate
from darts.metrics import rmse,  mase, mape,r2_score
from darts.dataprocessing.transformers import Mapper, Diff


ROOT_PATH = 'data'

df1 = etl_pipeline(extract_data_from_HS_json, file_path = f'{ROOT_PATH}/rent.json', cols = ['price_rent'], 
                            new_names=['Median Rent Price'],  
                            transform_dict = {"Median Rent Price" : growth(1)},
                            num_lags = 13, 
                            lagged_cols = ['Median Rent Price growth']
                            )
target = "Median Rent Price growth"

df_cov = df1.join(etl_pipeline(extract_data_from_StatsCanada_API, vector_id = 1235049756, latest_n = 240, new_name = "Unemployment rate",
                            transform_dict = {"Unemployment rate": [diff(1)],}
                            ))

df_cov = df_cov.join(etl_pipeline(extract_data_from_StatsCanada_API, vector_id = 111955499, latest_n = 240, new_name = "NHPI",
                                     transform_dict = {"NHPI": [growth(1)]}))

df_cov = df_cov.join(etl_pipeline(extract_data_from_HS_json, file_path = f'{ROOT_PATH}/rent.json', cols = ['rent_count', 'rent_listing_count'], 
                            new_names=['Total Leased', "New Listings"]))

df_cov["LNLR"] = df_cov["Total Leased"] / df_cov["New Listings"]
df_cov = make_lags(df_cov, 12, ["LNLR",])
df_cov = df_cov.ffill().bfill()
df_cov = df_cov[sorted(df_cov.columns)]


features = ["Unemployment rate diff", "NHPI growth", "LNLR", ] 

y = TimeSeries.from_dataframe(df_cov, value_cols=target)
y1 = TimeSeries.from_dataframe(df_cov, value_cols="Median Rent Price")

cov = TimeSeries.from_dataframe(df_cov, value_cols = features)

LR = LinearRegressionModel(    
    lags=12, 
    lags_future_covariates={"Unemployment rate diff": [-5],  "NHPI growth": [-3,-15], "LNLR": [-3],  },  
    output_chunk_length=1,)


LR.fit(y, future_covariates=cov)
forecast = LR.predict(3, y, future_covariates=cov)


def get_forecast(y, f, inverse_transform=True):
    if inverse_transform:
        to_exp = Mapper(lambda x: np.exp(x))
        f = to_exp.transform(f.cumsum()) * y.last_value()

    f = f.with_columns_renamed(target, "Median Rent Price")
    f = f.prepend_values([y.last_value()])
    y = y.pd_dataframe().reset_index()
    y['Data Type'] = 'Actual'
    f = f.pd_dataframe().reset_index()
    f['Data Type'] = 'Forecast'

    df = pd.concat([y,f], ignore_index=True, axis=0)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Median Rent Price"] = df["Median Rent Price"].astype(int)

    return df

df = get_forecast(y1, forecast)

df.to_csv(f'{ROOT_PATH}/forecast.csv', index=False)
df_cov.to_csv(f'{ROOT_PATH}/df_cov.csv')