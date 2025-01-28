import pandas as pd
import requests
import json

# def download_csv_from_StatsCanada(pid, data_path):
#     url = f"https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/{pid}/en"
#     response = requests.get(url)

#     # Check the response status
#     if response.status_code == 200:
#         # Parse the JSON response
#         result = response.json()
#         if result["status"] == "SUCCESS":
#             # Download the ZIP file
#             zip_response = requests.get(result["object"])
#             if zip_response.status_code == 200:
#                 # Extract the ZIP file
#                 with zipfile.ZipFile(io.BytesIO(zip_response.content)) as z:
#                     z.extractall(data_path) 
#                 print(f"CSV file extracted to {data_path}/{pid}.csv folder.")
#             else:
#                 print(f"Failed to download ZIP file. Status Code: {zip_response.status_code}")
#         else:
#             print(f"Failed to retrieve CSV URL. API Status: {result['status']}")
#     else:
#         print(f"Failed to connect to API. HTTP Status Code: {response.status_code}")

# tables = ["14100383", "18100205", "18100004"]
# for table in tables:
#     download_csv_from_StatsCanada(table, ROOT_PATH)

# Function to extract and process data from Statistics Canada CSV files
def extract_data_from_StatsCanada_CSV(file_path, in_col=None, out_cols=None, new_names=None, filters=None):
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
    df['Date'] = pd.to_datetime(['-'.join(x.split()[::-1]) for x in df['REF_DATE']])
    df.sort_values(by='Date', inplace=True)
    
    df_res = pd.DataFrame(index=df['Date'].drop_duplicates())

    for col in out_cols:
        df_ = df[df[in_col] == col] if in_col else df  # Filter based on in_col if provided
        
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
        df['Date'] = pd.to_datetime(df['Date'], format='%b %y')
    except ValueError:
        df['Date'] = pd.to_datetime(df['Date'], format='%y-%b') 
    df.set_index('Date', inplace=True)

    df = pd.DataFrame(df[in_col]).dropna()
    df.columns = new_names if new_names else in_col
    return df

def clean_integer(input_str):

    # Ensure input is a string, remove commas, strip any spaces
    cleaned_str = str(input_str).replace(',', '')
    
    # Convert the cleaned string to an integer
    return float(cleaned_str)

def extract_data_from_csv(file_path, in_col=None, new_name=None):

    df= pd.read_csv(file_path)
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
    except ValueError:
        df['Date'] = pd.to_datetime(df['Date'], format='%y-%b')
    df.set_index('Date', inplace=True)
    df = df[in_col].map(clean_integer).dropna()
    df.columns = (new_name if new_name else in_col)
    # print(df)
    return df

def extract_data_from_StatsCanada_API(vector_id, latest_n, new_name):
    """
    Fetches data from the API and converts it to a pandas DataFrame.

    Parameters:
    - vector_id (int): The vector ID for the data request.
    - latest_n (int): The number of latest periods to retrieve.

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
            raise ValueError("API response indicates failure: " + data[0].get("message", "Unknown error"))
        
        # Extract vector data points
        vector_data = data[0]["object"]["vectorDataPoint"]
        
        # Convert to DataFrame
        df = pd.DataFrame(vector_data)
        df = df.rename(columns={"refPer": "Date", "value": new_name})[["Date", new_name]]
        
        # df['Date'] = pd.to_datetime(['-'.join(x.split()[::-1]) for x in df['REF_DATE']])
        df['Date'] = pd.to_datetime(df['Date'])

        df.sort_values(by='Date', inplace=True)
        
        df = df.set_index('Date')
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
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract the chart data
    chart_data = data.get("data", {}).get("chart", [])
    
    if not chart_data:
        raise ValueError("No 'chart' data found in the JSON.")
    
    # Convert to a DataFrame
    df = pd.DataFrame(chart_data)
    
    # Ensure 'period' is included for Date conversion
    if 'period' not in cols:
        cols = ['period'] + cols
        new_names = ['Date'] + new_names
    
    # Filter the DataFrame and rename columns
    df = df[cols]
    df.columns = new_names
    
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
    df.set_index('Date', inplace=True)
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
    
    for col in lagged_cols:
        for lag in range(1, num_lags + 1):
            df[f"{col} lag {lag}"] = df[col].shift(lag)

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
def etl_pipeline(extraction_function, transform_dict=None, freq='ME', num_lags=18, lagged_cols=None, **kwargs):
    """
    Extracts, transforms, and processes data through a customizable ETL pipeline.

    Parameters:
    - extraction_function (function): The function used to extract data.
    - transform_dict (dict, optional): A dictionary mapping column names to transformations to apply.
    - freq (str, optional): Frequency for resampling the data (default is 'M' for monthly).
    - num_lags (int, optional): Number of lag periods to create (default is 12).
    - lagged_cols (list of str, optional): List of columns to create lags for. If None, all columns will be used.
    - kwargs: Additional arguments passed to the extraction function.

    Returns:
    - pd.DataFrame: The fully processed DataFrame after extraction, transformation, and lag creation.
    """
    # Step 1: Extract data using the provided extraction function
    extracted_df = extraction_function(**kwargs)
    
    # Step 2: Resample data to the specified frequency (e.g., monthly)
    transformed_df = extracted_df.sort_index().resample(freq).first()
    
    # Step 3: Apply transformations if provided
    if transform_dict:
        transformed_df = transform(transformed_df, transform_dict)
    # Step 4: Create lagged features if specified
    transformed_df = make_lags(transformed_df, num_lags, lagged_cols)
    
    return transformed_df

def growth(x):
    def inner(series):
        return series.pct_change(x) #np.log(series).diff(x)
    inner.__name__ = f"growth{x}" if x > 1 else "growth"
    return inner

def diff(x):
    def inner(series):
        return series.diff(x)
    inner.__name__ = f"diff{x}" if x > 1 else "diff"
    return inner

def infl_adjusted(infl):
    def inner(series):
        df_ = pd.concat([series, infl], axis = 1)
        return (df_.iloc[:,0] * df_.iloc[-1,1]/df_.iloc[:,1])
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