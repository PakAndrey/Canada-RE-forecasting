import pandas as pd
import requests
import json
import zipfile
import io
import numpy as np


def extract_from_RPS(file_path, in_col, sheet_name=0, new_name=None):
    """
    Reads an Excel file, skipping the first row and stopping at the first empty row.
    Optionally extracts a single column and renames it if specified.

    Parameters:
        filepath (str): Path to the Excel file.
        sheet_name (int or str, optional): Sheet name or index. Default is the first sheet.
        in_col (str): Column name to extract.
        new_name (str, optional): New name for the extracted column.

    Returns:
        pd.DataFrame or pd.Series: Processed DataFrame or Series with the required data.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=1)  # Skip first row

    # Find the first empty row (all NaN)
    first_empty_idx = df.isnull().all(axis=1).idxmax()

    # If no empty row found, return full dataframe, else return up to the first empty row
    df = df if df.isnull().all(axis=1).sum() == 0 else df.iloc[:first_empty_idx]

    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df.set_index("Date", inplace=True)

    # Extract specific column if provided
    df = df[[in_col]]
    if new_name:
        df.columns = [new_name]

    return df


def extract_data_from_StatsCanada_CSV(
    file_path, in_col=None, out_cols=None, new_names=None, filters=None
):
    """
    Extracts and processes data from a Statistics Canada CSV file.

    Parameters:
    - file_path (str): The name of the CSV file to read.
    - in_col (str, optional): Column to filter data by (e.g., a category column).
    - out_cols (list of str): List of columns to extract based on in_col filter.
    - new_names (list of str, optional): New names for the output columns.
    - filters (dict, optional): Additional filters to apply on the DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame containing the processed and filtered data.
    """
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(["-".join(x.split()[::-1]) for x in df["REF_DATE"]])

    df_res = pd.DataFrame(index=df["Date"].drop_duplicates())

    for col in out_cols:
        df_ = (
            df[df[in_col] == col] if in_col else df
        )  # Filter based on in_col if provided

        if filters:
            for filter_key, filter_value in filters.items():
                df_ = df_[df_[filter_key] == filter_value]

        df_ = df_[["VALUE", "Date"]].set_index("Date")
        df_res = pd.concat([df_res, df_], axis=1)

    df_res.columns = new_names if new_names else out_cols
    return df_res


# Function to extract and process data from CREA Excel files
def extract_data_from_CREA(file_path, sheet_name, in_col, new_names=None):
    """
    Extracts and processes data from a CREA Excel file.

    Parameters:
    - file_path (str): The name of the Excel file to read.
    - sheet_name (str): The sheet name within the Excel file to read.
    - in_col (str): Column name to extract from the sheet.
    - new_names (list of str, optional): New names for the output columns.

    Returns:
    - pd.DataFrame: A DataFrame containing the extracted and processed data.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%b %y")
    except ValueError:
        df["Date"] = pd.to_datetime(df["Date"], format="%y-%b")
    df.set_index("Date", inplace=True)

    df = pd.DataFrame(df[in_col]).dropna()
    df.columns = new_names if new_names else in_col
    return df


def clean_integer(input_str):

    # Ensure input is a string, remove commas, strip any spaces
    cleaned_str = str(input_str).replace(",", "")

    # Convert the cleaned string to an integer
    return float(cleaned_str)


def extract_data_from_csv(file_path, in_col=None, new_name=None):

    df = pd.read_csv(file_path)
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")
    except ValueError:
        df["Date"] = pd.to_datetime(df["Date"], format="%y-%b")
    df.set_index("Date", inplace=True)
    df = df[in_col].map(clean_integer).dropna()
    df.columns = new_name if new_name else in_col
    return df


def extract_data_from_StatsCanada_API(vector_id, latest_n, new_name):
    """
    Fetches data from the API and converts it to a pandas DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame containing the fetched data with columns "Date" and "Value".
    """
    # API endpoint
    url = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods"

    # Payload for the POST request
    post_body = [{"vectorId": vector_id, "latestN": latest_n}]

    try:
        # Send the POST request
        response = requests.post(url, json=post_body)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        data = response.json()

        # Ensure the response contains the expected structure
        if data[0]["status"] != "SUCCESS":
            raise ValueError(
                "API response indicates failure: "
                + data[0].get("message", "Unknown error")
            )

        # Extract vector data points
        vector_data = data[0]["object"]["vectorDataPoint"]

        # Convert to DataFrame
        df = pd.DataFrame(vector_data)
        df = df.rename(columns={"refPer": "Date", "value": new_name})[
            ["Date", new_name]
        ]

        df["Date"] = pd.to_datetime(df["Date"])

        df = df.set_index("Date")
        return df

    except requests.exceptions.RequestException as e:
        print(f"HTTP request failed: {e}")
        return None
    except ValueError as e:
        print(f"Data error: {e}")
        return None
    except KeyError as e:
        print(f"Unexpected data structure: {e}")
        return None


def extract_data_from_HS_json(file_path, cols, new_names):
    """
    Extract data from JSON, filter columns, rename them, and return a DataFrame.

    Parameters:
    - file_path (str): Path to the JSON file.
    - cols (list): List of column names to extract from the JSON.
    - new_names (list): New names for the extracted columns.

    Returns:
    - pd.DataFrame: Processed DataFrame with renamed columns.
    """
    # Load the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # Extract the chart data
    chart_data = data.get("data", {}).get("chart", [])

    if not chart_data:
        raise ValueError("No 'chart' data found in the JSON.")

    # Convert to a DataFrame
    df = pd.DataFrame(chart_data)

    # Ensure 'period' is included for Date conversion
    if "period" not in cols:
        cols = ["period"] + cols
        new_names = ["Date"] + new_names

    # Filter the DataFrame and rename columns
    df = df[cols]
    df.columns = new_names

    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
    df.set_index("Date", inplace=True)
    df = df.map(clean_integer)

    return df


# Function to create lagged versions of specified columns in a DataFrame
def make_lags(df, num_lags, lagged_cols=None):
    """
    Creates lagged versions of specified columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - num_lags (int): The number of lag periods to create.
    - lagged_cols (list of str, optional): List of columns to create lags for. If None, all columns will be used.

    Returns:
    - pd.DataFrame: The DataFrame with lagged columns added.
    """
    lagged_cols = lagged_cols or df.columns

    lag_dfs = [df] + [
        df[lagged_cols].shift(lag).add_suffix(f" lag {lag}")
        for lag in range(1, num_lags + 1)
    ]
    return pd.concat(lag_dfs, axis=1)

    return df


# Function to apply specified transformations to columns in a DataFrame
def transform(df, transform_dict):
    """
    Applies specified transformations to columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - transform_dict (dict): A dictionary mapping column names to transformation functions or lists of functions.

    Returns:
    - pd.DataFrame: The DataFrame with transformed columns added.
    """
    for col, funcs in transform_dict.items():
        if isinstance(funcs, list):
            for func in funcs:
                df[f"{col} {func.__name__}"] = pd.DataFrame(df[col]).apply(func)
        else:
            df[f"{col} {funcs.__name__}"] = pd.DataFrame(df[col]).apply(funcs)
    return df


# Main ETL pipeline function that combines extraction, transformation, and lag creation
def etl_pipeline(
    extractor,
    extractor_params,
    transform_dict=None,
    freq="ME",
    num_lags=18,
    lagged_cols=None,
) -> pd.DataFrame:
    """
    Extracts, transforms, and processes data through a customizable ETL pipeline.

    Returns:
    - pd.DataFrame: The fully processed DataFrame after extraction, transformation, and lag creation.
    """
    # Step 1: Extract data using the provided extraction function
    try:
        extracted_df = extractor(**extractor_params)
        if not isinstance(extracted_df.index, pd.DatetimeIndex):
            raise TypeError(f"DataFrame index must be a DatetimeIndex.")
    except TypeError as e:
        print(f"Extraction is not successful! {e}")

    # Step 2: Resample data to the specified frequency (e.g., monthly)
    transformed_df = extracted_df.sort_index().resample(freq).first()

    # Step 3: Apply transformations if provided
    try:
        if transform_dict:
            transformed_df = transform(transformed_df, transform_dict)
    except:
        print("Transformation is not successful!")

    # Step 4: Create lagged features if specified
    try:
        transformed_df = make_lags(transformed_df, num_lags, lagged_cols)
    except:
        print("Errors in lags!")

    return transformed_df


def growth(x):
    def inner(series):
        return np.log(series).diff(x)  # series.pct_change(x)

    inner.__name__ = f"growth{x}" if x > 1 else "growth"
    return inner


def diff(x):
    def inner(series):
        return series.diff(x)

    inner.__name__ = f"diff{x}" if x > 1 else "diff"
    return inner


def to_real(cpi, base_date=pd.Timestamp("20200131")):
    def inner(series):
        if base_date not in cpi.index:
            raise ValueError("Base year CPI not found in the CPI series.")
        if len(cpi.columns) > 1:
            raise ValueError("CPI has more than one column.")
        base_cpi = cpi.loc[base_date].values[0]
        return series * (base_cpi / cpi.loc[series.index].iloc[:, 0])

    inner.__name__ = f"real"
    return inner


def to_nominal(cpi, base_date=pd.Timestamp("20200131")):
    def inner(series):
        if base_date not in cpi.index:
            raise ValueError("Base year CPI not found in the CPI series.")
        base_cpi = cpi.loc[base_date].values[0]
        return series * (cpi.loc[series.index].iloc[:, 0] / base_cpi)

    inner.__name__ = f"real"
    return inner


def rolling_mean(x):
    def inner(series):
        return series.rolling(x).mean()

    inner.__name__ = f"mean{x}"
    return inner


def compose(f, g):
    def inner(x):
        return f(g(x))

    inner.__name__ = f"{f.__name__} {g.__name__}"
    return inner


def inv_growth(logdf, f0):
    return np.exp(logdf.cumsum()) * f0


def convert_forecast(y, f, inverse_transform=True):
    if inverse_transform:
        f = inv_growth(f, y.iloc[-1, 0]).astype(int)

    f.rename(
        columns={col: y.columns[i] for i, col in enumerate(f.columns)}, inplace=True
    )
    y["Data Type"] = "Actual"
    f["Data Type"] = "Forecast"
    df = pd.concat([y, f], axis=0).reset_index()
    print(df.tail(10).to_string())

    return df


# import yfinance as yf
# def extract_data_from_yfinance(ticker, new_names=None):
#     df = yf.download(ticker, start="2005-01-01", end="2025-01-17") #end="202-06-01"
#     etf = df.Close.resample('M').mean() #.ewm(alpha=0.1).mean()
#     # print(etf)

#     df = pd.DataFrame(etf)
#     df.columns = new_names if new_names else [ticker]
#     return df
