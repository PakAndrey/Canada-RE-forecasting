import numpy as np
import pandas as pd

# Function to extract and process data from Statistics Canada CSV files
def extract_data_from_StatsCanada(file_path, in_col=None, out_cols=None, new_names=None, filters=None):
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
    df_v = pd.read_excel(file_path, sheet_name=sheet_name)
    df_v['Date'] = pd.to_datetime(df_v['Date'], format='%b %y')
    df_v.set_index('Date', inplace=True)

    df_v = pd.DataFrame(df_v[in_col]).dropna()
    df_v.columns = new_names if new_names else in_col
    return df_v

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
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

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
                df[f"{col}_{func.__name__}"] = pd.DataFrame(df[col]).apply(func)
        else:
            df[f"{col}_{funcs.__name__}"] = pd.DataFrame(df[col]).apply(funcs)
    return df

# Main ETL pipeline function that combines extraction, transformation, and lag creation
def etl_pipeline(extraction_function, transform_dict=None, freq='MS', num_lags=13, lagged_cols=None, **kwargs):
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
        return np.log(series).diff(x)
    inner.__name__ = f"growth{x}"
    return inner

def diff(x):
    def inner(series):
        return series.diff(x)
    inner.__name__ = f"diff{x}"
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
    inner.__name__ = f"{f.__name__}_{g.__name__}"
    return inner