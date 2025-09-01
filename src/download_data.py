import requests
import json
import zipfile
from io import BytesIO
from datetime import datetime, timedelta
import os


DATA_PATH = "data"


def download_StatsCanada_data(pid, data_path):
    url = f"https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/{pid}/en"
    response = requests.get(url)

    # Check the response status
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()
        if result["status"] == "SUCCESS":
            # Download the ZIP file
            zip_response = requests.get(result["object"])
            if zip_response.status_code == 200:
                # Extract the ZIP file
                with zipfile.ZipFile(BytesIO(zip_response.content)) as z:
                    z.extractall(data_path)
                print(f"CSV file extracted to {data_path}/{pid}.csv folder.")
            else:
                print(
                    f"Failed to download ZIP file. Status Code: {zip_response.status_code}"
                )
        else:
            print(f"Failed to retrieve CSV URL. API Status: {result['status']}")
    else:
        print(f"Failed to connect to API. HTTP Status Code: {response.status_code}")


tables = [
    "34100145",
    "10100116",
    "18100256",
    "14100383",
    "18100205",
    # "14100287",
]  # ["14100383", "18100205", "18100004"]
for table in tables:
    download_StatsCanada_data(table, f"{DATA_PATH}/")


def fetch_HS_data(token, file_name):
    api_url = "https://housesigma.com/bkv2/api/stats/trend/chart"

    headers = {
        "accept": "application/json, text/plain, */*",
        "authorization": f"Bearer {token}",  # Use the token here
        "content-type": "application/json;charset=UTF-8",
        "hs-client-type": "desktop_v7",
        "hs-client-version": "7.21.77",
        "origin": "https://housesigma.com",
        "referer": "https://housesigma.com/on/market-trends/toronto-real-estate?municipality=10343&community=all&property_type=C.",
    }

    data = {
        "lang": "en_US",
        "province": "ON",
        "municipality": "10343",
        "community": "all",
        "house_type": "C.",
        "period_num": 180,
    }

    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        # Parse the response JSON data
        data = response.json()
        
        try:
            with open(file_name, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving data to {file_name}: {e}")
            print(data)
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")


fetch_HS_data("20250831cun49qi78j5ja29edg0r38pdhv", f"{DATA_PATH}/rent.json")


def generate_MLS_HPI_links():
    """Generate URLs for the current and previous month's MLS HPI data."""

    def format_link(date):
        month = date.strftime("%B")  # Full month name
        year = date.year
        return f"https://www.crea.ca/files/mls-hpi-data/MLS_HPI_{month}_{year}.zip"

    today = datetime.today()
    last_month = today.replace(day=1) - timedelta(days=1)  # Get previous month

    return [format_link(today), format_link(last_month)]


# print(generate_mls_hpi_links())


def download_MLS_HPI_data(extract_to: str):
    """
    Tries to download and extract the current month's MLS HPI ZIP file.
    If it fails, attempts to download the previous month's file.
    """
    os.makedirs(extract_to, exist_ok=True)

    for zip_url in generate_MLS_HPI_links():
        try:
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()  # Raise error if request fails

            with zipfile.ZipFile(BytesIO(response.content), "r") as zip_ref:
                zip_ref.extractall(extract_to)

            print(f"Successfully downloaded and extracted: {zip_url}")
            return  # Stop if successful

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {zip_url}: {e}")
        except zipfile.BadZipFile:
            print(f"Invalid ZIP file: {zip_url}")

    print("Both current and previous month's downloads failed.")


download_MLS_HPI_data(f"{DATA_PATH}/MLS_HPI/")
