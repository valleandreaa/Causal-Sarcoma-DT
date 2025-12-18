import os
import pandas as pd
from pymongo import MongoClient
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from transformers import pipeline
import sys
sys.path.append('/home/andrea/Desktop/Sarcoma-DT/ETL')
from etl_sdt.extract.excel_extractor import *
from etl_sdt.transform.data_transfomer import *
from etl_sdt.extract.pdf_extractor import *
from etl_sdt.transform.nlp_processor import *
from etl_sdt.config import renaming_dict, data_format_dict, relevant_features, free_text_columns
from etl_sdt.load.mongo_load import upsert_record  


# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 7, 27),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'etl_sarcoma_dt',
    default_args=default_args,
    description='An ETL process for Sarcoma data',
    schedule_interval=timedelta(days=1),
    max_active_runs=1,
    concurrency=1,
)




num_chunks = 3


def extract_data(num_chunks, **kwargs):
    df = pd.read_excel('data/adjumed_export_test.xlsx')
    # Group by patient_id
    grouped = df.groupby('Patient ID (PID)')
    
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
    
    # Convert each chunk to JSON
    chunk_jsons = [chunk.to_json() for chunk in chunks]
    print(f'Chunks: {len(chunk_jsons)}')
    return chunk_jsons

def transform_chunk_data(chunk_json, **kwargs):
    df = pd.read_json(chunk_json)

    trs = DictionaryTransformer(renaming_dict)
    df = trs.aggregate_dataframe(df)
    df = trs.renaming(df)
    df = trs.transform_dataframe(df, data_format_dict)

    fe = FeatureExtractor(df, relevant_features)
    df = fe.process()
    relevant_features_local = fe.get_relevant_features()

    ner = PathologyNERExtractor(model_name="alvaroalon2/biobert_genetic_ner")
    df = trs.report_info_extractor(df, ner)

    ner_genetic = PathologyNERExtractor(model_name="alvaroalon2/biobert_genetic_ner")
    ner_medical = PathologyNERExtractor(model_name="Clinical-AI-Apollo/Medical-NER")
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")

    df = trs.free_text_info_extractor(df, free_text_columns, translator, ner_genetic, ner_medical)
    df = trs.report_info_extractor(df, ner)

    records = trs.restructure_data(df, relevant_features_local, key_column='patient_id')
    
    return records

def load_data(records, **kwargs):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['SarcomaDB']
    collection = db['Patients']

    # Uncomment this to actually perform the upsert
    # for record in records:
    #     upsert_record(record, collection)

extract_data_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    op_kwargs={'num_chunks': num_chunks },
    dag=dag,
)


with TaskGroup('transform_chunk_group', dag=dag) as transform_chunk_group:
    
    
    for i in range(num_chunks ):  
        transform_chunk_task = PythonOperator(
            task_id=f'transform_chunk_{i}',
            python_callable=transform_chunk_data,
            op_kwargs={'chunk_json': '{{ ti.xcom_pull(task_ids="extract_data")[' + str(i) + '] }}'},
            provide_context=True,
            dag=dag,
        )

with TaskGroup('load_chunk_group', dag=dag) as load_chunk_group:
    
    for i in range(num_chunks ):  
        load_chunk_task = PythonOperator(
            task_id=f'load_chunk_{i}',
            python_callable=load_data,
            op_kwargs={'records': '{{ ti.xcom_pull(task_ids="transform_chunk_group.transform_chunk_' + str(i) + '") }}'},
            provide_context=True,
            dag=dag,
        )

# Define task dependencies
extract_data_task >> transform_chunk_group >> load_chunk_group
