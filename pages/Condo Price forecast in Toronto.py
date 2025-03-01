import streamlit as st
import pandas as pd

from src.dashboard import *

FORECAST_PATH = "forecasts"


df = pd.read_csv(f"{FORECAST_PATH}/forecast_hpi_toronto.csv")
df["Date"] = pd.to_datetime(df["Date"]).dt.date

df_cov = pd.read_csv(f"{FORECAST_PATH}/df_cov.csv")
df_cov["Date"] = pd.to_datetime(df_cov["Date"])
df_cov.set_index("Date", inplace=True)

target = "Toronto MLS Price Benchmark"
features = ["Mortgage rate 5y log", "SNLR", "Active Listings growth"]


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title="Canadian Housing Market Trends",
    page_icon=":flag-ca:",  # This is an emoji shortcode. Could be a URL too.
    layout="wide",
)


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
"""
# Forecasting Condo Price Benchmark

The Condo Price Benchmark is a key measure of condominium prices in a region, designed to track the value of typical condo units over time. It is based on the repeat sales methodology, which analyzes price changes of the same properties rather than relying on averages or medians. This approach provides a more accurate and stable representation of condo price trends by filtering out distortions caused by shifts in the mix of units sold. 
"""

# Add some spacing
""
""

st.header("MLS Condo Price Benchmark in Toronto", divider="gray")


""
left, right = st.columns([2, 1])
with left:
    show_forecast(df, target)
with right:
    st.dataframe(df[df["Data Type"] == "Forecast"][["Date", target]], hide_index=True)

" "
" "

st.header(target + " and Leading Indicators", divider="gray")

"""
	•	**Mortgage Rate 5Y Log**: The logarithmic transformation of the conventional mortgage lending rate for a 5-year term, as reported by the Canada Mortgage and Housing Corporation (CMHC). This helps analyze long-term borrowing cost trends while smoothing fluctuations.

	•	**SNLR (Sales-to-New Listings Ratio)**: A key housing market indicator that measures the balance between supply and demand by comparing home sales to new listings. Higher values indicate a seller’s market, while lower values suggest a buyer’s market.
    
	•	**Active Listings growth**: refers to the percentage change in the number of properties listed for sale in a given market over a specific period. It indicates whether housing inventory is increasing or decreasing, helping to assess market supply and demand dynamics.
"""

selected_feature = st.selectbox("Choose a leading indicator:", features)

# Lag Selection
max_lag = max(
    int(col.split("lag")[1].strip()) if "lag" in col else 0
    for col in df_cov.columns
    if selected_feature in col
)
selected_lag = st.slider("Choose a lag value:", 0, max_lag, 0)

left, right = st.columns([1, 2])

with left:
    # Generate Plot
    plot_target_covariates(df_cov, selected_feature, selected_lag, f"{target} growth")


with right:
    plot_line_chart(df_cov, selected_feature, selected_lag, f"{target} growth")
