import streamlit as st
import pandas as pd

from src.dashboard import *

FORECAST_PATH = "forecasts"


df = pd.read_csv(f"{FORECAST_PATH}/forecast_hpi.csv")
df["Date"] = pd.to_datetime(df["Date"]).dt.date

df_cov = pd.read_csv(f"{FORECAST_PATH}/df_cov.csv")
df_cov["Date"] = pd.to_datetime(df_cov["Date"])
df_cov.set_index("Date", inplace=True)

target = "MLS Price Benchmark"
features = ["Mortgage rate 5y log", "M2++ growth"]


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
# Forecasting Home Price Benchmark

The Home Price Benchmark is a key measure of housing prices in a region, designed to track the value of typical homes across the country. It is based on repeat sales methodology, which analyzes price changes of the same properties over time rather than relying on averages or medians. This approach helps provide a more accurate and stable representation of home price trends, filtering out distortions caused by shifts in the mix of properties sold. The benchmark is widely used by financial institutions, policymakers, and analysts to assess market conditions and forecast housing trends.
"""

# Add some spacing
""
""

st.header("RPS National Home Price Benchmark in Canada", divider="gray")


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

	•	**M2++ growth**: Measures the growth rate of the M2++ money supply, which includes cash, chequing, savings deposits, and broader liquid assets. It reflects the availability of money in the economy, impacting inflation and housing demand.
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
