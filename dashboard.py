import streamlit as st
import subprocess
import os
from src.dashboard import *


FORECAST_PATH = "forecasts"


# GitHub repo details
GITHUB_REPO = "PakAndrey/Canada-RE-forecasting"
BRANCH = "forecasts-branch"


def fetch_latest_forecasts():
    """Fetch the latest forecasts from the private branch."""
    if not os.path.exists("forecasts"):
        os.makedirs("forecasts")

    # Clone only the forecasts branch
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            BRANCH,
            f"https://github.com/{GITHUB_REPO}.git",
            "forecast_repo",
        ],
        check=True,
    )

    # Move forecasts from cloned repo
    os.system("mv forecast_repo/forecasts/* forecasts/")
    os.system("rm -rf forecast_repo")  # Clean up


fetch_latest_forecasts()


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title="Canadian Housing Market Trends",
    page_icon=":flag-ca:",  # This is an emoji shortcode. Could be a URL too.
)


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
"""
# Forecasting Canadian Housing Market Trends

The Canadian housing market has been a topic of much discussion, particularly 
in recent years due to fluctuating economic conditions. This page focuses 
on forecasting various property market trends such as Toronto's median condo rent prices 
and the National Home Price Benchmark in Canada.
"""

""
""


df = load_forecast(f"{FORECAST_PATH}/forecast_hpi.csv")
st.header("RPS National Home Price Benchmark in Canada", divider="gray")
show_forecast(df, "MLS Price Benchmark")

""
df = load_forecast(f"{FORECAST_PATH}/forecast_hpi_toronto.csv")
st.header("MLS Condo Price Benchmark in Toronto", divider="gray")
show_forecast(df, "Toronto MLS Price Benchmark")

""

df = load_forecast(f"{FORECAST_PATH}/forecast.csv")
st.header("Median Condo Rental Price in Toronto", divider="gray")
show_forecast(df, "Median Rent Price")
