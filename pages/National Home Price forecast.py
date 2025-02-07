import streamlit as st
import pandas as pd

from src.dashboard import *

FORECAST_PATH = 'forecasts'


df = pd.read_csv(f'{FORECAST_PATH}/forecast_hpi.csv')
df["Date"] = pd.to_datetime(df["Date"]).dt.date

df_cov = pd.read_csv(f'{FORECAST_PATH}/df_cov_hpi.csv')
df_cov["Date"] = pd.to_datetime(df_cov["Date"])
df_cov.set_index("Date", inplace=True)

target = "RPS Home Price Benchmark"
features = ["Mortgage rate 5y diff", "M2++ growth", "Sales"] 


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
# Forecasting Home Price Benchmark

The Home Price Benchmark is a key measure of housing prices in a region, designed to track the value of typical homes across the country. It is based on repeat sales methodology, which analyzes price changes of the same properties over time rather than relying on averages or medians. This approach helps provide a more accurate and stable representation of home price trends, filtering out distortions caused by shifts in the mix of properties sold. The benchmark is widely used by financial institutions, policymakers, and analysts to assess market conditions and forecast housing trends.
'''

# Add some spacing
''
''

st.header('RPS National Home Price Benchmark in Canada', divider='gray')


''
left, right = st.columns([2,1])
with left:
    show_forecast(df, target)
with right:
    st.dataframe(df[df['Data Type'] == 'Forecast'][['Date', target]].iloc[1:], hide_index=True)

' '
' '

st.header(target +  " and Leading Indicators", divider="gray")

'''
	•   **Mortgage Rate 5Y diff**: The monthly change in the conventional mortgage lending rate for a 5-year term, as reported by the Canada Mortgage and Housing Corporation (CMHC). This rate is a key benchmark for mortgage affordability and borrowing costs in Canada.

	•	**M2++ growth**: Measures the growth rate of the M2++ money supply, which includes cash, chequing, savings deposits, and broader liquid assets. It reflects the availability of money in the economy, impacting inflation and housing demand.

	•	**Sales**: The total number of residential properties sold within a month, based on MLS (Multiple Listing Service) data. It serves as a key indicator of housing market activity and demand.
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
    # Generate Plot
    plot_target_covariates(df_cov, selected_feature, selected_lag, "HPI growth")


with right:
    plot_line_chart(df_cov, selected_feature, selected_lag, 'HPI growth')