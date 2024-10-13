# Forecasting Canadian Real Estate Trends

The Canadian housing market has garnered significant attention, especially in recent years due to shifting economic conditions. This repository is dedicated to forecasting key real estate trends, including Toronto's median rent prices and the MLS Composite Home Price Benchmark, which tracks the price trajectory of a typical home across Canada.

## Overview

- data/ :                           Directory containing datasets
- src/ETL.py :                         Scripts for data extraction, transformation, and loading
- src/backtesting_forecasting.py :     Scripts for forecasting and evaluating predictive models
- HPI_forecasting.ipynb :             Jupyter notebook for exploration, analysis, modelling and forecasting of MLS Composite Home Price Benchmark in Canada
- Toronto_rent_forecasting.ipynb :    Jupyter notebook for exploration, analysis, modelling and forecasting of Toronto's median rent prices
- README.md :                         Project overview and instructions
- requirements.txt :                   Python package dependencies


## Data Sources

The data used in this project is sourced from the following:

- Statistics Canada: Provides economic indicators such as money supply (M2++) and mortgage rates, etc.
- Canadian Real Estate Association (CREA): Provides monthly home sales and MLS Composite Home Price Benchmark data.