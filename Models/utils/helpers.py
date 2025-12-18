from pymongo import MongoClient, errors
from bson import datetime as bson_datetime
import pandas as pd
import yaml 
from tabulate import tabulate
from pandas import json_normalize
from collections.abc import MutableMapping
import re
from datetime import datetime

class MongoExtractor:
    def __init__(self, connection_string, database_name, collection_name, config_file=None, custom_config=None):
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = MongoClient(self.connection_string, socketTimeoutMS=200000, connectTimeoutMS=200000)
        self.db = self.client[self.database_name]
        self.collection = self.db[self.collection_name]
        if custom_config:
            self.config = {'data_config': custom_config}
        elif config_file:
            self.config = self.load_config(config_file)
        else:
            self.config = {}

    @staticmethod
    def load_config(config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def flatten_dict(self, d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and k != 'patient_info':
                for i, item in enumerate(v):
                    if isinstance(item, MutableMapping):
                        items.extend(self.flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                    else:
                        items.append((f"{new_key}{sep}{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)

    def flatten_dict_episode(self, d, parent_key='', sep='.'):
        """
        Flatten dictionary for episode data where episodes are already unwound.
        Handles nested dictionaries and lists but avoids numerical indices for episode arrays.
        """
        items = []
        for k, v in d.items():
            
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, MutableMapping):
                items.extend(self.flatten_dict_episode(v, new_key, sep=sep).items())

            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, MutableMapping):
                        items.extend(self.flatten_dict_episode(item, new_key, sep=sep).items())
                    else:
                        # For primitive values in lists, just use the key directly
                        items.append((new_key, item))
            else:
                # Simple key-value assignment
                items.append((new_key, v))
        return dict(items)

    def flatten_data(self, data):
        return [self.flatten_dict_episode(item) for item in data]


    def get_patient_by_id(self, patient_id):
        data = self.fetch_data_patient_id(patient_id)
        return pd.DataFrame(data)
    
    @staticmethod
    def get_current_date():
        from datetime import datetime
        return datetime.utcnow()

    def replace_placeholder(self, d, placeholder, value):
        if isinstance(d, dict):
            return {k: self.replace_placeholder(v, placeholder, value) for k, v in d.items()}
        elif isinstance(d, list):
            return [self.replace_placeholder(i, placeholder, value) for i in d]
        elif isinstance(d, str) and placeholder in d:
            return value
        return d

    def fetch_data(self):
        aggregator = self.config['data_config'].get('aggregator', None)
        
        try:

            cursor = self.collection.aggregate(aggregator)

            data = list(cursor)
            flattened_data = self.flatten_data(data)
            return flattened_data

        except errors.PyMongoError as e:
            print(f"An error occurred: {e}")
            return []

    def fetch_data_patient_id(self, patient_id):
        aggregator = self.config['data_config'].get('aggregator', None)
        if aggregator is None:
            aggregator = []
        
        # Create a deep copy to avoid modifying the original config
        import copy
        aggregator = copy.deepcopy(aggregator)

        # Modify any $project stage to wrap field references with $ifNull to handle missing fields
        for stage in aggregator:
            if '$project' in stage and isinstance(stage['$project'], dict):
                updated_project = {}
                for field, value in stage['$project'].items():
                    if isinstance(value, str) and value.startswith('$'):
                        # Wrap field references with $ifNull to return null for missing fields
                        updated_project[field] = {"$ifNull": [value, None]}
                    else:
                        updated_project[field] = value
                stage['$project'] = updated_project

        # Insert a match stage to filter by patient_id.
        match_stage = {"$match": {"_id": patient_id}}
        aggregator.insert(0, match_stage)
        
        try:
            cursor = self.collection.aggregate(aggregator)
            data = list(cursor)
            flattened_data = self.flatten_data(data)
            return flattened_data

        except errors.PyMongoError as e:
            print(f"An error occurred: {e}")
            return []

    def fetch_data_config(self, name_config: str='data_config'):
        fields = self.config[name_config]['fields']
        filters = self.config[name_config]['filters']
        
        query = filters if filters else {}
        projection = {}

        if fields:
            for field in fields:
                projection[field] = 1

        try:
            cursor = self.collection.find(query, projection)
            data = list(cursor)
            flattened_data = self.flatten_data(data)
            return flattened_data

        except errors.PyMongoError as e:
            print(f"An error occurred: {e}")
            return []

    def get_dataframe(self):
        data = self.fetch_data()
        return pd.DataFrame(data)

    @staticmethod
    def reorder_dataframe_columns(df: pd.DataFrame, feature_spec: dict) -> pd.DataFrame:
        """
        Reorder DataFrame columns to match the order specified in feature_spec.
        
        This is critical because MongoDB returns columns in alphabetical order,
        but our encoding/decoding logic assumes columns are in the same order
        as specified in the YAML feature specification.
        
        Args:
            df: DataFrame with potentially misordered columns
            feature_spec: Dictionary with feature names as keys (from YAML config)
            
        Returns:
            DataFrame with columns reordered to match feature_spec order
        """
        ordered_columns = []
        
        # First, add columns in the order they appear in feature_spec
        for feature_name in feature_spec.keys():
            if feature_name in df.columns:
                ordered_columns.append(feature_name)
            else:
                print(f"Warning: Feature '{feature_name}' from spec not found in DataFrame columns")
        
        # Add any extra columns that might be in df but not in feature_spec (like _id)
        for col in df.columns:
            if col not in ordered_columns:
                ordered_columns.append(col)
        
        # Reorder the DataFrame
        reordered_df = df[ordered_columns]
        print(f"Reordered DataFrame columns to match feature_spec order")
        print(f"Original column order: {list(df.columns)}")
        print(f"New column order: {list(reordered_df.columns)}")
        
        return reordered_df




    def get_dataframe_os(self):
        data = self.fetch_data()
        df = pd.DataFrame(data)
        
        # Convert date columns to datetime
        df["outcomes.date_last_patient_contact_information"] = pd.to_datetime(
            df["outcomes.date_last_patient_contact_information"]
        )
        df["outcomes.date_of_initial_diagnosis"] = pd.to_datetime(
            df["outcomes.date_of_initial_diagnosis"]
        )
        
        df['days_to_treatment'] = (df['treatment_features.date'] - df['outcomes.date_of_initial_diagnosis']).dt.days.abs()

        # Calculate the time difference in years for status 3
        df["time_difference_years"] = (
            (df["outcomes.date_last_patient_contact_information"] - df["outcomes.date_of_initial_diagnosis"])
            .dt.days / 365.25
        )
        
        # Calculate the time difference in years from today for status 1 and 2
        today = datetime.now()
        df["time_difference_from_today_years"] = (
            (today - df["outcomes.date_of_initial_diagnosis"]).dt.days / 365.25
        )
        
        # Categorize based on the difference
        def categorize_survival(row):
            if row["outcomes.status_last_patient_contact_information"] == 3:
                if row["time_difference_years"] < 3:
                    return 0
                elif 3 <= row["time_difference_years"] < 5:
                    return 1
                elif 5 <= row["time_difference_years"] < 10:
                    return 2
                else:
                    return 3
            elif row["outcomes.status_last_patient_contact_information"] in [1, 2]:
                if row["time_difference_from_today_years"] < 3:
                    return 0
                elif 3 <= row["time_difference_from_today_years"] < 5:
                    return 1
                elif 5 <= row["time_difference_from_today_years"] < 10:
                    return 2
                else:
                    return 3
            return None

        df["category_overall_survival"] = df.apply(categorize_survival, axis=1)

        return df


    def get_dataframe_first_treatments(self):
        data = self.fetch_data()
        return pd.DataFrame(data)

    def get_dataframe_markers(self):
        data = self.fetch_data()
        df = pd.DataFrame(data)
        biomarkers = ["MDM2", "EWSR1", "CD34", "BRAF", "ALK1"]
        biomarker_columns = [col for col in df.columns if col.startswith("pattern_expressions")]

        # Create a binary DataFrame to track the presence of each biomarker
        binary_biomarkers_df = pd.DataFrame(0, index=df.index, columns=biomarkers)

        # Define a function to clean and extract biomarkers
        def extract_biomarkers(value):
            # Remove brackets and quotes, and strip whitespace
            cleaned_value = re.sub(r"[\[\]']", "", str(value)).strip()
            # Split by commas if multiple biomarkers are combined in the same cell
            return cleaned_value.split(", ")

        # Iterate over each biomarker column to populate the binary matrix
        for col in biomarker_columns:
            column_values = df[col].dropna()

            for biomarker in biomarkers:
                # Apply extraction and check for biomarker presence
                binary_biomarkers_df[biomarker] = binary_biomarkers_df[biomarker] | column_values.apply(
                    lambda x: 1 if biomarker in extract_biomarkers(x) else 0
                )

        # Ensure binary format (0 or 1) for each biomarker column
        binary_biomarkers_df = binary_biomarkers_df.astype(int)

        # Remove original biomarker columns and concatenate the cleaned binary columns
        df_cleaned = df.drop(columns=biomarker_columns)
        df_combined = pd.concat([df_cleaned, binary_biomarkers_df], axis=1)
        
        return df_combined

    def get_dataframe_treatment(self, name_config: str='data_config'):
        data_1 = self.fetch_data_config(name_config)
        return pd.DataFrame(data_1)

    def print_patient_count(self):
        filters = self.config['data_config']['filters']
        count = self.collection.count_documents(filters)
        table = [["Filter", "Count"]]
        for key, value in filters.items():
            table.append([f"{key}: {value}", count])
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

    def close_connection(self):
        self.client.close()

class BenchmarkingExtractor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, sep='\t')
        self.df.to_csv('output.csv', sep=',', index=False)

    def extract_benchmarking_data(self):
        self.df['metastasis_initial_presentation_exists'] = self.df.apply(
                lambda row: 1 if row['Metastasis_nature2012'] == 'M1' else 0, axis=1
            )
        
        self.df['patient_status'] = self.df.apply(
            lambda row: 'Deceased' if not pd.isna(row['Days_to_date_of_Death_nature2012']) 
            else 'Alive' if not pd.isna(row['Days_to_Date_of_Last_Contact_nature2012']) 
            else 'Unknown', axis=1
        )

        self.df['status_code'] = self.df['patient_status'].apply(lambda status: 0 if status == 'Alive' else 1)
        self.df['region_code'] = 0

        benchmarking_columns = {
            'sampleID': '_id',
            'Age_at_Initial_Pathologic_Diagnosis_nature2012': 'patient_info.age_at_admission',
            'Gender_nature2012': 'patient_info.gender',
            'CN_Clusters_nature2012': 'patient_info.age_class',
            'Converted_Stage_nature2012': 'doc_1.diagnosis.diagnosis_malignant_benign',
            'region_code': 'doc_1.diagnosis.anatomic_region_code_grouping',
            'metastasis_initial_presentation_exists': 'status.metastasis_initial_presentation_exists',
            'status_code': 'status.status_last_patient_contact_information'
        }

        extracted_df = self.df[list(benchmarking_columns.keys())].rename(columns=benchmarking_columns)
        extracted_df = extracted_df.dropna()
        extracted_df.to_csv('output.csv', sep=',', index=False)
        return extracted_df