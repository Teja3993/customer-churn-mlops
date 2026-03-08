import os
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

# Configure logging for MLOps tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data():
    # The exact Kaggle dataset slug
    dataset_name = "beatafaron/telco-customer-churn-realistic-customer-feedback"
    download_path = "data/raw"

    try:
        logging.info("Authenticating with Kaggle API...")
        api = KaggleApi()
        api.authenticate()

        logging.info(f"Downloading dataset: '{dataset_name}' to '{download_path}'...")
        
        # Download and automatically unzip the CSVs directly into the raw data folder
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        
        logging.info(f"Success! Dataset downloaded and extracted to {download_path}/")
        
    except Exception as e:
        logging.error(f"Failed to download dataset. Ensure kaggle.json is in your C:\\Users\\<Username>\\.kaggle\\ directory.")
        logging.error(f"System Error: {e}")

if __name__ == "__main__":
    # Ensure the destination directory exists before downloading
    os.makedirs("data/raw", exist_ok=True)
    fetch_data()