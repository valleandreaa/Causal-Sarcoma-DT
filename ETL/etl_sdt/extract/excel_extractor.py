import pandas as pd
import os
import re
from datetime import datetime
from etl_sdt.utils.logging_config import logger 
import numpy as np

def merge_x_y_columns(df):
    """
    Merge columns with _x and _y suffixes by combining valid values with '|'.
    
    Args:
    - df (pd.DataFrame): DataFrame with potential _x/_y duplicate columns
    
    Returns:
    - pd.DataFrame: DataFrame with merged columns
    """
    # Find all _x columns
    x_cols = [col for col in df.columns if col.endswith('_x')]
    
    for x_col in x_cols:
        base_col = x_col[:-2]  # Remove '_x' suffix
        y_col = f"{base_col}_y"
        
        if y_col in df.columns:
            # Merge the two columns
            df[base_col] = df.apply(
                lambda row: (
                    f"{row[x_col]}|{row[y_col]}"
                    if pd.notna(row[x_col]) and pd.notna(row[y_col])
                    else (row[x_col] if pd.notna(row[x_col]) else row[y_col])
                ),
                axis=1
            )
            # Drop the _x and _y columns
            df = df.drop(columns=[x_col, y_col])
    
    return df


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

def merge_dataframes_on_id(df_main, df_radio, df_syst, key_column="PID"):
    """
    Build a long event table:
      - df_radio: already clean radiotherapy table (one row per RT episode)
      - df_syst : already clean systemic therapy table (one row per cycle/episode)
      - df_main : patient table that contains extra radiotherapy '(1. Teil)' columns
                 that may be pipe-separated and must be exploded into multiple events

    Returns:
      - df_events_merged: events table merged with patient-level columns
    """

    # -----------------------------
    # local helpers
    # -----------------------------
    def _split_pipe_cell(x):
        if pd.isna(x):
            return []
        s = str(x).strip()
        if not s:
            return []
        return [p.strip() for p in re.split(r"\s*\|\s*", s) if p.strip()]

    def _explode_pipe_columns(df, pipe_cols):
        """
        Explode rows where any pipe_cols contain lists (split by '|').
        Align by index; broadcast singletons; fill mismatch with NaN.
        """
        df = df.copy()

        for col in pipe_cols:
            if col in df.columns:
                df[col] = df[col].apply(_split_pipe_cell)

        rows = []
        for _, r in df.iterrows():
            lengths = [len(r[c]) for c in pipe_cols if c in df.columns]
            max_len = max(lengths) if lengths else 0

            if max_len == 0:
                rows.append(r.to_dict())
                continue

            for i in range(max_len):
                rr = r.to_dict()
                for c in pipe_cols:
                    if c not in df.columns:
                        continue
                    vals = rr[c]
                    if isinstance(vals, list):
                        if len(vals) == 0:
                            rr[c] = np.nan
                        elif len(vals) == 1:
                            rr[c] = vals[0]  # broadcast
                        elif i < len(vals):
                            rr[c] = vals[i]
                        else:
                            rr[c] = np.nan
                rows.append(rr)

        return pd.DataFrame(rows)

    # -----------------------------
    # build events from clean tables
    # -----------------------------
    df_radio_events = df_radio.assign(
        event_type="radiotherapy",
        event_date=df_radio["Start Radiotherapy"],
        event_start_date=df_radio["Start Radiotherapy"],
        event_end_date=df_radio.get("End Radiotherapy"),
    )

    df_syst_events = df_syst.assign(
        event_type="systemic_therapy",
        event_date=df_syst["Start date of cycle"],
        event_start_date=df_syst["Start date of cycle"],
        event_end_date=df_syst.get("End date of cycle"),
    )

    events_df = pd.concat([df_radio_events.rename(columns={'PID': "Patient ID (PID)"}), 
                           df_syst_events.rename(columns={'PID': "Patient ID (PID)"})], 
                          ignore_index=True)

    # -----------------------------
    # extract & unnest RT "(1. Teil)" from df_main
    # -----------------------------
    teil_cols = [
        "Start Radiotherapy (1. Teil)",
        "End Radiotherapy (1. Teil)",
        "Indication for radiotherapy (1. Teil)",
        "Type of radiotherapy (1. Teil)",
        "Number of fractions given (1. Teil)",
        "Total dose used in radiotherapy (1. Teil)",
        # optional extras you listed
        "redonc_first_indication_Timo",
        "redonc_first_type_Timo",
        "redonc_fractions_gray_Timo",
        "Hyperthermia status & interventionell radiology",
    ]

    existing_teil_cols = [c for c in teil_cols if c in df_main.columns]

    if existing_teil_cols:
        # choose "key fields" that indicate Teil-1 is present
        key_fields = [c for c in [
            "Start Radiotherapy (1. Teil)",
            "Total dose used in radiotherapy (1. Teil)",
            "Number of fractions given (1. Teil)",
        ] if c in df_main.columns]

        if key_fields:
            teil_mask = df_main[key_fields].notna().any(axis=1)
        else:
            # fallback: any teil col present
            teil_mask = df_main[existing_teil_cols].notna().any(axis=1)

        df_teil = df_main.loc[teil_mask, [key_column] + existing_teil_cols].copy()

        # explode pipe-separated values (at least start/end/fractions/dose)
        df_teil = _explode_pipe_columns(
            df_teil,
            pipe_cols=[c for c in [
                "Start Radiotherapy (1. Teil)",
                "End Radiotherapy (1. Teil)",
                "Number of fractions given (1. Teil)",
                "Total dose used in radiotherapy (1. Teil)",
                "Type of radiotherapy (1. Teil)",
                "Indication for radiotherapy (1. Teil)",
            ] if c in df_teil.columns],
        )

        df_teil_events = df_teil.assign(
            event_type="radiotherapy_first",
            event_date=df_teil.get("Start Radiotherapy (1. Teil)"),
            event_start_date=df_teil.get("Start Radiotherapy (1. Teil)"),
            event_end_date=df_teil.get("End Radiotherapy (1. Teil)"),
        )

        events_df = pd.concat([events_df, df_teil_events], ignore_index=True)


    # -----------------------------
    # merge events with main patient table
    # -----------------------------
    # (use df_main PID column name on the right; you said df_main has Patient ID (PID))
    if "patient_id" in df_main.columns:
        df_events_merged = events_df.merge(
            df_main,
            left_on=key_column,
            right_on="patient_id",
            how="left",
        )
    else:
        df_main = df_main.rename(
            columns={"Total dose used in radiotherapy": "radiotherapy_total_dose_overall"}
        )
        df_events_merged = events_df.merge(
            df_main,
            left_on=key_column,
            right_on="Patient ID (PID)",
            how="left",
        )


    # your existing cleanup
    df_events_merged = merge_x_y_columns(df_events_merged)

    return df_events_merged
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
        
        # Merge any overlapping columns with _x and _y suffixes
        merged_df = merge_x_y_columns(merged_df)
        
        logger.info("Data extraction and merging completed successfully.")
        return merged_df
    except Exception as e:
        logger.error(f"Error occurred while extracting data from Excel file: {e}")
        return None

def open_excel(file_path, sheet_name=None):
    """
    Opens an Excel file and returns a DataFrame.

    Args:
    - file_path (str): The path to the Excel file.

    Returns:
    - pd.DataFrame: The data as a pandas DataFrame.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
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
