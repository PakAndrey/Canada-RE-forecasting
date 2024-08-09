# Canadian Home Price Benchmark Prediction 

This repository focuses on predicting the MLS Composite Home Price Benchmark in Canada, which represents the price trajectory of a typical home in the country. The objective is to identify leading economic indicators and build a forecasting model that provides a 6-month ahead prediction of home prices.

## Overview

The Canadian housing market has been a topic of much discussion, particularly in recent years due to fluctuating economic conditions. This project aims to leverage key economic indicators such as mortgage rates and money supply (M2++) to forecast the national home price benchmark with a 6-month horizon.

- data/ :                           Directory containing datasets
- src/ETL.py :                         Scripts for data extraction, transformation, and loading
- src/backtesting_forecasting.py :     Scripts for forecasting and evaluating predictive models
- HPI_forecasting.ipynb :             Jupyter notebook for exploration, analysis, modelling and forecasting
- README.md :                         Project overview and instructions
- requirements.txt :                   Python package dependencies


## Data Sources

The data used in this project is sourced from the following:

- Statistics Canada: Provides economic indicators such as money supply (M2++) and mortgage rates, etc.
- Canadian Real Estate Association (CREA): Provides monthly home sales and MLS Composite Home Price Benchmark data.