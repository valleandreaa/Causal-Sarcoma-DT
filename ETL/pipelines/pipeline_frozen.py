import argparse
import pandas as pd
from pymongo import MongoClient
import os
import sys
from dotenv import load_dotenv
import numpy as np

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from etl_sdt.extract.excel_extractor import open_excel, chunking, merge_dataframes_on_id
from etl_sdt.configs.config_frozen import *
from etl_sdt.utils.logging_config import logger
from etl_sdt.transform.data_transfomer import DictionaryTransformer, FeatureExtractor


def process_file(file_path, collection, preview=False, preview_n=5, timeline=False):
    """
    Main processing logic for the ETL pipeline.
    """
    df_main = open_excel(file_path=file_path, sheet_name='MDS_KSW_LUKS')
    logger.info(f"File {file_path} loaded successfully.")
    
    df_rt = open_excel(file_path=file_path, sheet_name='MDS_RT (2. Teil)')
    df_syst = open_excel(file_path=file_path, sheet_name='MDS_SYST')

    df = merge_dataframes_on_id(df_main, df_rt, df_syst, key_column='Patient ID (PID)')

    chunks = chunking(df, num_chunks=50, chunking_variable_id='Patient ID (PID)')

    for _, df_chunk in enumerate(chunks):
        # chunk_json = chunk.to_json(orient='records')
        # df_chunk = DictionaryTransformer.read_chuck(chunk_json)

        dict_transf = DictionaryTransformer(column_renaming_map)
        # df_chunk = dict_transf.aggregate_dataframe(df_chunk)
        df_chunk = dict_transf.renaming(df_chunk)
        df_chunk = dict_transf.transform_nested_values(df_chunk)
        df_chunk = dict_transf.transform_dataframe(df_chunk, data_format_dict=data_type_mapping)
        df_chunk = dict_transf.map_columns_with_partial_labels(df_chunk, label_dict)
        fe = FeatureExtractor(df_chunk, feature_mapping, data_type_mapping)
        df_chunk = fe.process_frozen()

        df_chunk = fe.post_process_frozen()

        records = dict_transf.restructure_to_records(df_chunk, feature_mapping, key_column='patient_id', timeline=timeline)

        if preview:
            logger.info("Preview of processed records:")
            for rec in records[:preview_n]:
                logger.info(rec)
            continue  # Skip storing if preview is enabled

        for record in records:
            upsert_record(record, collection)  # Store each record immediately


def convert_numpy_types(record):
    """
    Recursively convert numpy data types in a record to native Python types.
    """
    if isinstance(record, dict):
        return {key: convert_numpy_types(value) for key, value in record.items()}
    elif isinstance(record, list):
        return [convert_numpy_types(item) for item in record]
    elif pd.isna(record):  # Handle NaN, NaT, and None
        return None
    elif isinstance(record, (np.integer, np.floating)):
        return record.item()
    elif isinstance(record, np.ndarray):
        return record.tolist()
    else:
        return record


def upsert_record(record, collection):
    # try:
        record = convert_numpy_types(record)  # Convert numpy types before upserting
        query = {"_id": record["_id"]}
        update = {"$set": record}
        result = collection.update_one(query, update, upsert=True)

        if result.upserted_id:
            logger.info(f"Inserted new document with _id: {result.upserted_id}")
        else:
            logger.info(f"Updated existing document with _id: {record['_id']}")
    # except Exception as e:
    #     logger.error(f"Failed to upsert record with _id: {record['_id']}. Error: {e}")


def main(file_path, preview=False, dry_run=False, preview_n=5, delete_collection=False, timeline=False):
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../', '.env'))

    client = MongoClient(os.getenv('MONGO_URI'))
    db = client[os.getenv('MONGO_DB')]
    collection = db[os.getenv('MONGO_COLLECTION')]

    if delete_collection:
        logger.info("Dropping collection before processing...")
            

    process_file(file_path, collection=collection, preview=preview, preview_n=preview_n, timeline=timeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL Pipeline for Medical Data")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the Excel file')
    parser.add_argument('--preview', action='store_true', help='Preview transformed records (does not insert)')
    parser.add_argument('--dry_run', action='store_true', help='Transform data and show what would be inserted (no DB write)')
    parser.add_argument('--preview_n', type=int, default=5, help='Number of records to preview if --preview is used')
    parser.add_argument('--delete_collection', action='store_true', help='Delete the collection before processing')
    parser.add_argument('--timeline', action='store_true', help='Use 6-month timeline-based episodes instead of 30-day threshold episodes. Episodes extend to cover follow-up period. Intermediary gap-filled episodes copy diagnosis with original status, last gap-filled episode includes only minimal final status diagnosis.')
    
    args = parser.parse_args()

    main(
        file_path=args.file_path,
        preview=args.preview,
        dry_run=args.dry_run,
        preview_n=args.preview_n,
        delete_collection=args.delete_collection,
        timeline=args.timeline
    )