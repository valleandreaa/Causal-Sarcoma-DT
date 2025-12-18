import os 
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.getcwd(), 'ETL'))
from etl_sdt.extract.excel_extractor import *
from etl_sdt.transform.data_transfomer import *
from etl_sdt.extract.pdf_extractor import *
from etl_sdt.transform.nlp_processor import *
from etl_sdt.config import *
# from transformers import pipeline


# import df 
df = extract_excel('ETL/data/adjumed_export_20241027-120926.xlsx')


# Filter out rows based on Patient ID (PID)
# df= df[~df['Patient ID (PID)'].isin(patient_ids_to_filter)]

chunks = chunking(df, num_chunks= 50)
ner_genetic = PathologyNERExtractor(model_name="alvaroalon2/biobert_genetic_ner") 

# ner_medical = PathologyNERExtractor(model_name="Clinical-AI-Apollo/Medical-NER") 

# translator = pipeline("translation", model="t5-small")

# Loop over each chunk and process it
for i, chunk in enumerate(chunks):
    
    chunk_json = chunk.to_json(orient='records')
    
    
    df = DictionaryTransformer.read_chuck(chunk_json)
    
    
    
    trs = DictionaryTransformer(renaming_dict)
    df = trs.aggregate_dataframe(df)
    df = trs.renaming(df)
    df = trs.transform_dataframe(df, data_format_dict)
    
    
    fe = FeatureExtractor(df, relevant_features)

    df = fe.process()

    df = fe.post_process()
    relevant_features = fe.get_relevant_features()
    
    # Print or save the processed DataFrame
    # print(f"Processed Chunk {i+1}:\n{df}\n")

    # ***
    # df = trs.free_text_info_extractor(df, free_text_columns,  ner_genetic)

    # df = trs.free_text_info_extractor(df, free_text_columns, translator,  ner_genetic,  ner_medical)


    # TO ADD NER
    # ***
    # df = trs.report_info_extractor(df, ner_genetic)



    records = trs.restructure_data(df, relevant_features, key_column= 'patient_id')
    # records = trs.process_restructured_data(records)





    from pymongo import MongoClient
    from datetime import datetime

    # Establish a connection to MongoDB
    client = MongoClient(os.getenv('MONGO_URI'))

    # Select the database
    db = client[os.getenv('MONGO_DB')]
    # Select the collection
    collection = db[os.getenv('MONGO_COLLECTION')]

    #TODO: add env variables into read me

    def upsert_record(record):
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
            
            # Update documents
            result = collection.update_one(query, update, upsert=True)
            
            if result.upserted_id:
                print(f"Inserted new document with _id: {result.upserted_id}")
            else:
                print(f"Updated existing document with _id: {record['_id']}")
        except Exception as e:
            print(f"Failed to upsert record with _id: {record['_id']}. Error: {e}")




    for record in records: 
        upsert_record(record)



