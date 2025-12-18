from pymongo import MongoClient
from datetime import datetime
# from utils.logging_config import logger
# Establish a connection to MongoDB



def upsert_record(record, collection):
    """
    Update the record if it exists, or insert it if it doesn't.
    
    Args:
    record (dict): The record to be inserted or updated.
    """
    try:
        # Define the query to find the document
        query = {"_id": record["_id"]}
        
        # Define the update operation
        update = {"$set": record}
        
        # Update the document if it exists, insert if it doesn't
        result = collection.update_one(query, update, upsert=True)
        
        if result.upserted_id:
            print(f"Inserted new document with _id: {result.upserted_id}")
        else:
            print(f"Updated existing document with _id: {record['_id']}")
    except Exception as e:
        print(f"Failed to upsert record with _id: {record['_id']}. Error: {e}")




