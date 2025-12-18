import pandas as pd
import os
import re
from datetime import datetime
from etl_sdt.utils.logging_config import logger 


def get_latest_file(directory, prefix, extension):
    """
    Get the latest file in the directory that matches the given prefix and extension.

    Args:
    - directory (str): The path to the directory to search.
    - prefix (str): The prefix of the file names to match.
    - extension (str): The extension of the file names to match.

    Returns:
    - str: The name of the latest file.
    """

    logger.info(f"Searching for the latest file in directory: {directory} with prefix: {prefix} and extension: {extension}")

    # Regular expression to match the file pattern
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{8}}-\d{{6}})\.{re.escape(extension)}")

    latest_file = None
    latest_time = None

    for filename in os.listdir(os.getcwd() + directory):
        match = pattern.match(filename)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
            if latest_time is None or timestamp > latest_time:
                latest_time = timestamp
                latest_file = filename

    logger.info(f"Latest file found: {latest_file}")
    return latest_file

def extract_excel(file_path):
    """
    Extracts and merges data from all sheets in an Excel file based on a key column.

    Args:
    - file_path (str): The path to the Excel file.
    - key_column (str): The column to use as the key for merging sheets.

    Returns:
    - pd.DataFrame: The merged data as a pandas DataFrame.
    """
    try:
        # Read all sheets
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        df_list = []

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df_list.append(df)

        # Merge all sheets into a single DataFrame based on the key column
        # merged_df = pd.concat(df_list, ignore_index=True)
        key_column = list(set(df_list[0]).intersection(df_list[1].columns))
        merged_df = pd.merge(df_list[0], df_list[1], on=key_column, how='inner')
        
        logger.info("Data extraction and merging completed successfully.")
        return merged_df
    except Exception as e:
        logger.error(f"Error occurred while extracting data from Excel file: {e}")
        return None
    
def open_excel(file_path):
    """
    Opens an Excel file and returns a DataFrame.

    Args:
    - file_path (str): The path to the Excel file.

    Returns:
    - pd.DataFrame: The data as a pandas DataFrame.
    """
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        logger.info("Excel file opened successfully.")
        return df
    except Exception as e:
        logger.error(f"Error occurred while opening Excel file: {e}")
        return None

def chunking(df: pd.DataFrame, num_chunks: int =10, chunking_variable_id: str = 'Patient ID (PID)' ):
    grouped = df.groupby(chunking_variable_id)
    
    # Calculate chunk size based on the number of groups and num_chunks
    num_groups = len(grouped)
    chunk_size = max(1, num_groups // num_chunks)
    
    # Create a list of DataFrames, each containing multiple patients
    chunks = []
    for i, (patient_id, group) in enumerate(grouped):
        if i % chunk_size == 0:
            chunks.append(group)
        else:
            chunks[-1] = pd.concat([chunks[-1], group])
    
    return chunks

def features_filter(df, relevant_features):
    return df[relevant_features]


class patient_EHR:
    def __init__(self, rename_dict):
        self.EHRr = {}
        self.rename_dict= rename_dict

    def __print__(self, patient_id):
        pass

    def extract_general_caracteristics(self):
        pass

    def appender_interventions(self):
        pass

    

    def assign_general_charasteristics(self, df, key_column):
        records = []
        for _, row in df.iterrows():
            patient_record = row.to_dict()
            patient_id = patient_record.pop(key_column)
            records.append({"_id": patient_id, "data": patient_record})
        return records
