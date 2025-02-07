import streamlit as st
import pandas as pd

from src.dashboard import *

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression

FORECAST_PATH = 'forecasts'


df = pd.read_csv(f'{FORECAST_PATH}/forecast.csv')
df["Date"] = pd.to_datetime(df["Date"]).dt.date

df_cov = pd.read_csv(f'{FORECAST_PATH}/df_cov.csv')
df_cov["Date"] = pd.to_datetime(df_cov["Date"])
df_cov.set_index("Date", inplace=True)

target = "Median Rent Price growth"
features = ["Unemployment rate diff", "NHPI growth", "LNLR"] 
target2 = "Median Rent Price"

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Canadian Housing Market Trends',
    page_icon=':flag-ca:', # This is an emoji shortcode. Could be a URL too.
    layout="wide",
)


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# Forecasting Median Rental Price

Median Condo Rent Price refers to the midpoint rental price of condominiums in a given market. This means that half of the listed rental prices are above this value, and half are below. Unlike averages, the median rent price is less affected by extreme high or low rental listings, making it a more reliable indicator of typical rent costs in a specific city or region. It helps renters, investors, and policymakers understand housing affordability and market trends over time.
'''

# Add some spacing
''
''

st.header('Median Condo Rental Price in Toronto', divider='gray')

''

left, right = st.columns([2,1])
with left:
    show_forecast(df, target2)
with right:
    st.dataframe(df[df['Data Type'] == 'Forecast'][['Date', target2]].iloc[1:], hide_index=True)




st.header("Rent Price and Leading Indicators", divider="gray")

'''
	•	**Unemployment Rate Diff**: The change in the unemployment rate over a given period, indicating shifts in labor market conditions. A rising unemployment rate may signal economic downturns, while a decline suggests job market improvement.

	•	**NHPI Growth**: The growth rate of the New Housing Price Index (NHPI), which tracks changes in the prices of new residential homes in Canada. This metric reflects trends in housing affordability and construction costs.
    
	•	**LNLR (Leased to New Listings Ratio)**: A measure of rental market tightness, calculated as the ratio of leased properties to newly listed rental units. A higher LNLR suggests strong demand for rentals, while a lower ratio indicates a more balanced or oversupplied market.
'''

selected_feature = st.selectbox("Choose a leading indicator:", features)

# Lag Selection
max_lag = max(
    int(col.split("lag")[1].strip()) if "lag" in col else 0
    for col in df_cov.columns if selected_feature in col
)
selected_lag = st.slider("Choose a lag value:", 0, max_lag, 0)


left, right = st.columns([1,2])
with left:
    plot_target_covariates(df_cov, selected_feature, selected_lag, target)
with right:
    plot_line_chart(df_cov, selected_feature, selected_lag, target)
