import streamlit as st
import pandas as pd

from src.dashboard import *


ROOT_PATH = 'data'


df = pd.read_csv(f'{ROOT_PATH}/forecast.csv')
df["Date"] = pd.to_datetime(df["Date"]).dt.date

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Canadian Housing Market Trends',
    page_icon=':flag-ca:', # This is an emoji shortcode. Could be a URL too.
)


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# Forecasting Canadian Housing Market Trends

The Canadian housing market has been a topic of much discussion, particularly 
in recent years due to fluctuating economic conditions. This page focuses 
on forecasting various property market trends such as Toronto's median condo rent prices 
and the National Home Price Benchmark in Canada.
'''

''
''


''
st.header('Median Condo Rental Price in Toronto', divider='gray')
show_forecast(df, "Median Rent Price")

''

df = pd.read_csv(f'{ROOT_PATH}/forecast_hpi.csv')
df["Date"] = pd.to_datetime(df["Date"]).dt.date

st.header('RPS National Home Price Benchmark in Canada', divider='gray')
show_forecast(df, "RPS Home Price Benchmark")