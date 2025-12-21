import pandas as pd
import numpy as np
import re 
from collections import defaultdict 
import warnings
import json
import os 
from etl_sdt.extract.pdf_extractor import extract_text_from_pdf
import unicodedata
# from langdetect import detect
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Any, List, Dict, Union
from etl_sdt.utils.logging_config import logger
# import langid
from sklearn.model_selection import train_test_split
from math import ceil
import itertools
import collections.abc
from io import StringIO

class DictionaryTransformer:
    def __init__(self, rename_dict, key_mappings=None, keys_to_remove=None):
        self.key_mappings = key_mappings if key_mappings else {}
        self.keys_to_remove = keys_to_remove if keys_to_remove else []
        self.rename_dict = rename_dict if rename_dict else []
    
    @staticmethod
    def read_chuck(chunk_json):
        return pd.read_json(StringIO(chunk_json))

    def rename_keys(self, input_dict):
        return {self.key_mappings.get(key, key): value for key, value in input_dict.items()}

    def remove_keys(self, input_dict):
        return {key: value for key, value in input_dict.items() if key not in self.keys_to_remove}

    def transform_values(self, input_dict):
        transformed_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, pd.Series):
                value = value.drop_duplicates().values[0] if not value.empty else np.nan
            transformed_dict[key] = str(value) if isinstance(value, (int, float)) and not np.isnan(value) else value
        return transformed_dict
    
    def map_columns_with_partial_labels(self, df, label_dict):
        def has_token_overlap(val, partial):
            val_tokens = set(str(val).lower().split())
            partial_tokens = set(str(partial).lower().split())
            return not val_tokens.isdisjoint(partial_tokens)

        for col, config in label_dict.items():
            mapping = config["mapping"]
            mode = config.get("mode", "substring")

            def match(val):
                val_str = str(val).lower()
                for partial, mapped in mapping.items():
                    partial_str = str(partial).lower()
                    if (
                        (mode == "substring" and partial_str in val_str) or
                        (mode == "token" and has_token_overlap(val_str, partial_str))
                    ):
                        return mapped
                return None

            df[col] = df[col].apply(match)
        return df
    def get_column_names(self, df):
        """
        Returns the column names of the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame.

        Returns:
            list: A list of column names.
        """
        logger.info("Getting column names from DataFrame")
        return df.columns.tolist()

    def clean_nested_dict(self, nested_dict):
        """
        Recursively cleans a nested dictionary.

        Args:
            nested_dict (dict): The nested dictionary to clean.

        Returns:
            dict: The cleaned dictionary.
        """
        logger.info("Cleaning nested dictionary")
        cleaned_dict = {}
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                cleaned_dict[key] = self.clean_nested_dict(value)
            else:
                cleaned_dict[key] = value
        return cleaned_dict

    def transform(self, input_list):
        """
        Transforms a list of dictionaries by cleaning nested dictionaries and renaming/removing keys.

        Args:
            input_list (list): The list of dictionaries to transform.

        Returns:
            list: The transformed list of dictionaries.
        """
        logger.info("Transforming input list")
        transformed_list = []
        for item in input_list:
            transformed_item = {}
            for key, value in item.items():
                if isinstance(value, dict):
                    cleaned_value = self.clean_nested_dict(value)
                else:
                    cleaned_value = value
                transformed_item[key] = cleaned_value
            transformed_list.append(self.rename_keys(self.remove_keys(transformed_item)))
        logger.info("Transformation completed")
        return transformed_list

    def find_keys_with_substring(self, input_dict, substring):
        """
        Finds keys in a dictionary that contain a given substring.

        Args:
            input_dict (dict): The dictionary to search.
            substring (str): The substring to search for.

        Returns:
            list: A list of matched keys.
        """
        logger.info(f"Finding keys with substring '{substring}'")

        matched_keys = []
        for key, value in input_dict.items():
            if substring in key:
                matched_keys.append(key)
            if isinstance(value, dict):
                matched_keys.extend(self.find_keys_with_substring(value, substring))
        
        logger.info(f"Found {len(matched_keys)} keys with substring '{substring}'")
        return matched_keys

    def aggregate_columns_with_same_base(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Groups columns with the same base name.

        Args:
            columns (List[str]): A list of column names.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are base names and values are lists of columns with that base name.
        """
        logger.info("Aggregating columns with the same base name")
        # Compile regular expressions to match column names with and without parentheses
        pattern_with_parens = re.compile(r'^(.*) (\(\d+\)|\[\d+\])$')
        pattern_without_parens = re.compile(r'^(.*)$')

        # Initialize a defaultdict to store grouped columns
        column_groups = defaultdict(list)
        
        # Iterate through each column name and match it against the regular expressions
        for col in columns:
            match_with_parens = pattern_with_parens.match(col)
            match_without_parens = pattern_without_parens.match(col)
            if match_with_parens:
                base_name = match_with_parens.group(1)
                column_groups[base_name].append(col)
            elif match_without_parens:
                base_name = match_without_parens.group(1)
                column_groups[base_name].append(col)
        logger.info("Aggregation completed")
        return column_groups
    
    def _expand_lists(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expands lists within DataFrame columns into separate rows.

        Args:
            df (pd.DataFrame): The input DataFrame with list columns.

        Returns:
            pd.DataFrame: The DataFrame with expanded lists.
        """
        for col in df.columns:
            # Check if the column has lists
            if df[col].apply(lambda x: isinstance(x, list)).all():
                df[col] = df[col].apply(self._expand_list)
        return df

    def _expand_list(self, lst: List[Any]) -> List[Any]:
        """
        Expands a list by splitting strings containing '|' and flattening the result.

        Args:
            lst (List[Any]): The input list to be expanded.

        Returns:
            List[Any]: The expanded list.
        """
        expanded_list = []
        for item in lst:
            if isinstance(item, str) and '|' in item:
                expanded_list.extend(item.split('|'))
            else:
                expanded_list.append(item)
        return expanded_list

    def _replace_empty_lists_with_none(self, df):
        """
        Replace all empty lists in a DataFrame with None.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: A DataFrame with empty lists replaced by None.
        """
        return df.applymap(lambda x: None if isinstance(x, list) and len(x) == 0 else x)
    
    def create_aggregated_data(self, df: pd.DataFrame, grouped_columns: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Creates a DataFrame to store aggregated results by grouping columns and aggregating values row-wise.

        Args:
            df (pd.DataFrame): The input DataFrame.
            grouped_columns (Dict[str, List[str]]): A dictionary where keys are base names and values are lists of columns with that base name.

        Returns:
            pd.DataFrame: A DataFrame where columns are base names and values are aggregated lists.
        """
        aggregated_data = {}

        # Iterate through each group and aggregate the values row-wise
        for base_name, group in grouped_columns.items():
            aggregated_data[base_name] = df[group].apply(lambda row: list(row.dropna()), axis=1)

        # Create a new DataFrame from the aggregated data
        aggregated_df = pd.DataFrame(aggregated_data)

        return aggregated_df

    def aggregate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates the DataFrame by grouping columns with the same base name and 
        applying various transformations.

        Args:
            df (pd.DataFrame): The input DataFrame to be aggregated.

        Returns:
            pd.DataFrame: The aggregated DataFrame with expanded lists and unique values.
        """
        # Get the column names from the DataFrame
        flat_features = self.get_column_names(df)
        
        # Group columns with the same base name
        grouped_columns = self.aggregate_columns_with_same_base(flat_features)
        
        # Create a new dictionary to store aggregated results
        aggregated_df = self.create_aggregated_data(df, grouped_columns)

        # Expand lists within the DataFrame
        aggregated_df = self._expand_lists(aggregated_df)
        
        # Extract unique values from the DataFrame
        # Really need it?
        df = aggregated_df.applymap(self._extract_unique_values)
        
        # Replace empty lists with None
        df = self._replace_empty_lists_with_none(df)
        
        return df

    def _extract_unique_values(self, nested_list):
        unique_values = set()
        
        def flatten_and_extract(lst):
            if isinstance(lst, list):
                for item in lst:
                    if isinstance(item, list):
                        flatten_and_extract(item)
                    else:
                        unique_values.add(item)
        
        flatten_and_extract(nested_list)
        
        if len(unique_values) == 1:
            return unique_values.pop()  
        else:
            return list(unique_values)

    def _flatten_features(self, relevant_features):
        return [item for sublist in relevant_features.values() for item in sublist]

    def _filter_relevanat_fetaures(self, df, flat_features):
        # Ensure all relevant features are in the DataFrame
        flat_features = [feature for feature in flat_features if feature in df.columns]
        return df[flat_features]

    def _rename_columns(self, df, rename_dict):
        return df.rename(columns=rename_dict)

    def _key_allocation(self, group_data, row, feature):

        renamed_feature = self.rename_dict.get(feature, feature)
        try:
            group_data[renamed_feature] = row[renamed_feature]
        except KeyError:
            warnings.warn(f"Failed to allocate key: {renamed_feature}")

    def renaming(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames the columns of the DataFrame based on the rename_dict attribute.

        Args:
            df (pd.DataFrame): The input DataFrame to be renamed.

        Returns:
            pd.DataFrame: The DataFrame with renamed columns.
        """
        return df.rename(columns=self.rename_dict)
    
    def _sort_by_date_columns(self, group_df, ascending=True):
        # Iterate through columns to find date columns
        exclude_words = ["initial", "lowest", "highest", "entry", "newest", "oldest", "latest", "earliest", "first", "last"]
        for column in group_df.columns:
            if pd.api.types.is_datetime64_any_dtype(group_df[column]) and not any(word in column for word in exclude_words):
                # Sort by the date column
                group_df = group_df.sort_values(by=column, ascending=ascending)
                # break
        return group_df
    
    def _sort_by_column(self, group_df, column, ascending=True): 
        # Check if any value in the column is a list
        # if any(isinstance(value, list) for value in group_df[column]):
        #     print(f"List found in column '{column}', skipping sorting for this group.")
        #     return group_df
        
        # Otherwise, sort the group_df by the column
        return group_df.sort_values(by=column, ascending=ascending)
    
    def _filter_column_name(self, df: pd.DataFrame, patterns: List[str]) -> pd.DataFrame:
        """
        Filters columns in the DataFrame based on the given patterns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            patterns (List[str]): A list of patterns to filter columns.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        filtered_columns = [col for col in df.columns if all(pattern in col for pattern in patterns)]
        return df[filtered_columns]

    def _get_pdf_files(self, value: Union[str, List[str]], folder_path: str) -> List[str]:
        """
        Retrieves the file paths of PDF files from the given value and folder path.

        Args:
            value (Union[str, List[str]]): The value to check for PDF files.
            folder_path (str): The folder path to search for PDF files.

        Returns:
            List[str]: A list of file paths of PDF files.
        """
        file_paths = []
        if isinstance(value, str) and value.lower().endswith('.pdf'):
            return [self._check_file_in_folder(value, folder_path)]
            
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.lower().endswith('.pdf'):
                    file_path = self._check_file_in_folder(item, folder_path)
                    file_paths.append(file_path)
            return file_paths
        else:
            return file_paths

    def _rename_column(self, base: str, appender: str) -> str:
        """
        Renames a column by appending an appender string to the base name.

        Args:
            base (str): The base name of the column.
            appender (str): The string to append to the base name.

        Returns:
            str: The renamed column name.
        """
        return f'{base}_{appender}'

    def _check_file_in_folder(self, file_name: str, folder_path: str) -> str:
        """
        Checks if a file exists in the given folder path.

        Args:
            file_name (str): The name of the file to check.
            folder_path (str): The path of the folder to search for the file.

        Returns:
            str: The full path of the file if it exists, None otherwise.
        """
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            return file_path
        return None
    
    def _create_missing_columns(self, df: pd.DataFrame, columns_report: List[str]) -> pd.DataFrame:
        """
        Creates missing columns in the DataFrame for annotations.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns_report (List[str]): A list of column names containing 'report' or 'upload'.

        Returns:
            pd.DataFrame: The DataFrame with missing columns created.
        """
        new_columns = {}
        for cl_name in columns_report:
            new_column_name = self._rename_column(cl_name, 'ner')
            if new_column_name not in df.columns:
                new_columns[new_column_name] = None

        if new_columns:
            df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
        
        return df
    
    def _merge_dicts(self, main: dict, new: dict) -> None:
        """
        Merges two dictionaries recursively.

        Args:
            main (dict): The main dictionary to merge into.
            new (dict): The new dictionary to merge.

        Returns:
            None
        """
        for key, value in new.items():
            if key in main:
                if isinstance(main[key], dict) and isinstance(value, dict):
                    self._merge_dicts(main[key], value)
                elif isinstance(main[key], list) and isinstance(value, list):
                    main[key].extend(value)
                else:
                    main[key] = value  # Overwrite if it's not a list or dict
            else:
                main[key] = value


    def report_info_extractor(self, df: pd.DataFrame, *args) -> pd.DataFrame:
        """
        Extracts information from free text columns in the DataFrame and adds annotations.

        Args:
            df (pd.DataFrame): The input DataFrame.
            *args: Additional arguments for entity extraction.

        Returns:
            pd.DataFrame: The DataFrame with added annotations.
        """
        # Filter columns containing 'report' or 'upload'
        columns_report = self._filter_column_name(df, ['report', 'upload'])
        
        # Create missing columns for annotations
        df = self._create_missing_columns(df, columns_report)
        
        def process_files(files):
            annotations = {}
            for pdf_path in files:
                pdf_text = extract_text_from_pdf(pdf_path, detect_vertical_text=False, word_margin=0.1)
                if pdf_text:
                    for arg in args:
                        entities = arg.extract_entities(pdf_text)
                        if entities:
                            ner_tags = arg.extract_entities_dynamic(entities)
                        else:
                            ner_tags = {}
                        self._merge_dicts(annotations, ner_tags)
            return self.unique_values_in_dict(annotations) if annotations else None



        def extract_annotations(row):
            annotations_dict = {}
            for cl_name in columns_report:
                files = self._get_pdf_files(row[cl_name], folder_path='./ETL/data/pdfs')
                annotations = process_files(files)
                if annotations:
                    annotations = annotations['genetic'] if 'genetic' in annotations else annotations
                annotations_dict[self._rename_column(cl_name, 'ner')] = annotations if annotations else None
            return pd.Series(annotations_dict)

        
        annotations_df = df.apply(extract_annotations, axis=1)
        df.update(annotations_df)
        
        return df

    def unique_values_in_dict(self, d):
        """
        This function takes a dictionary and replaces any lists in the dictionary
        with a list of unique values.
        
        :param d: Dictionary to process
        :return: Dictionary with lists replaced by lists of unique values
        """
        # for key, value in d.items():
        if isinstance(d, list):
            d = list(set(d))
        return d

    def normalize_unicode(self, text):
        """
        Normalizes unicode characters in the given text by converting them to ASCII.

        Args:
            text (str or List[str]): The input text or list of texts to normalize.

        Returns:
            str or List[str]: The normalized text or list of normalized texts.
        """
        if isinstance(text, list):
            # Normalize each text in the list
            return [unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('ascii') for t in text]
        else:
            # Normalize the single text
            normalized_text = unicodedata.normalize('NFKD', text)
            ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
            return ascii_text
        
    def detect_and_translate(self, text: str, translator) -> str:
        """
        Detects the language of the text and translates it if it is in German.

        Args:
            text (str): The input text to detect and translate.
            translator: The translation function.

        Returns:
            str: The translated text if it is in German, otherwise a message indicating the text is not in German or an error occurred.
        """
        try:
            # Detect the language of the text
            # language = detect(text)
            language, _ = langid.classify(text)
            # print(f"Detected language: {language}")

            if language == 'de':                 
                translation = translator(text, max_length=2000)
                return translation[0]['translation_text']
            else:
                return "The text is not in German."

        except Exception as e:
            return f"An error occurred: {str(e)}"

    def free_text_info_extractor(self, df: pd.DataFrame, free_text_columns: List[str], *args) -> pd.DataFrame:
        """
        Extracts information from free text columns in the DataFrame and adds annotations.

        Args:
            df (pd.DataFrame): The input DataFrame.
            free_text_columns (List[str]): The list of column names containing free text.
            translator: The translation function.
            *args: Additional arguments for entity extraction.

        Returns:
            pd.DataFrame: The DataFrame with added annotations.
        """
        df = self._create_missing_columns(df, free_text_columns)

        def preprocess_and_extract_annotations(free_text):
            annotations = {}
            if free_text:
                # free_text = self.detect_and_translate(self.normalize_unicode(free_text), translator)
                for arg in args:
                    try:
                        entities = arg.extract_entities(free_text)
                        ner_tags = arg.extract_entities_dynamic(entities) if entities else {}
                        self._merge_dicts(annotations, ner_tags)
                        if annotations:
                            annotations = annotations['genetic'] if 'genetic' in annotations else annotations
                    except Exception as e:
                        # Handle the exception (e.g., log it, raise a custom exception, etc.)
                        print(f"An error occurred: {e}")
            return [self.unique_values_in_dict(annotations)] if annotations else None

        for cl_name in free_text_columns:
            df[self._rename_column(cl_name, 'ner')] = df[cl_name].apply(preprocess_and_extract_annotations)

        return df


    def _rename_column_ner(self, df, column_name):
        new_column_name = f'{column_name}_ner'
        df[new_column_name] = None
        return df, new_column_name
        

    def rename_list(self, features):
        return [self.rename_dict.get(feature, feature) for feature in features]
    
    def extract_columns(self, row: pd.Series, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        extracted_data = {}
        for key, value in feature_dict.items():
            if isinstance(value, list):
                extracted_data[key] = {
                    sub_key: row.get(sub_value)
                    for sub_dict in value
                    for sub_key, sub_value in sub_dict.items()
                    if pd.notnull(row.get(sub_value))
                }
            elif isinstance(value, dict):
                extracted_data[key] = self.extract_columns(row, value)
            else:
  

                value = self.rename_dict.get(value, value)
                val = row.get(value)
                if not isinstance(val, list) :
                    if pd.notnull(val):
                        extracted_data[key] = val


                elif isinstance(val, list) and ('_ner' in value or 'substance' in value):
                    if pd.notnull(val).any():
                        extracted_data[key] = val

        return extracted_data
    


    def restructure_data(self, df: pd.DataFrame, relevant_features: dict, key_column: str) -> list:
        """
        Restructures the data from the dataframe into a list of patient records.

        Parameters:
        df (pd.DataFrame): The input dataframe containing the data.
        relevant_features (dict): A dictionary where keys are group names and values are lists of features.
        key_column (str): The column name to group by, typically a patient identifier.

        Returns:
        list: A list of dictionaries, each representing a patient record.
        """

        records = []
        key_column = 'patient_id'  
        for group_idx, (patient_id, group_df) in enumerate(df.groupby(key_column), start=1):
            
            patient_record = {"_id": patient_id}
            
            row = group_df.iloc[-1]
            patient_record['outcomes'] = self.extract_columns(row, relevant_features['outcomes_final'])

            patient_record['initial_diagnosis'] = self.extract_columns(row, relevant_features['initial_diagnosis'])

            patient_record['status_metrics'] = self.extract_columns(row, relevant_features['status_metrics'])

            patient_record['patient_info'] = self.extract_columns(row, relevant_features['patient_info'])


            group_df = self._sort_by_column(group_df, column=['patient_id', 'time_relative_sarcomaboard_presentation'], ascending=True)
                        
            patient_record['diagnosis_treatment_sequence'] = []
            for idx, (index, row) in enumerate(group_df.iterrows(), start=1):
                             
                for group, features in relevant_features.items():
                    renamed_features = self.rename_list(features)

                    def get_date_existance(relevant_features, group):
                        date_key = relevant_features[group].get('date')
                        new_key = self.rename_dict.get(date_key)
                        return new_key
                    
                    if group == 'radiology':
                        if row[get_date_existance(relevant_features, group)]:
                            dict_tmp = self.extract_columns(row, relevant_features[group])
                            patient_record['diagnosis_treatment_sequence'].append(dict_tmp)


                    if group == 'biopsy' :
                        if not pd.isna(row[get_date_existance(relevant_features, group)]):
                            dict_tmp = self.extract_columns(row, relevant_features[group])
                            patient_record['diagnosis_treatment_sequence'].append(dict_tmp)
                    if group == 'surgery' :
                        if not pd.isna(row[get_date_existance(relevant_features, group)]):
                            dict_tmp = self.extract_columns(row, relevant_features[group])
                            patient_record['diagnosis_treatment_sequence'].append(dict_tmp)

                    if group == 'chemotherapy' :
                        if not pd.isna(row[get_date_existance(relevant_features, group)]):
                            dict_tmp = self.extract_columns(row, relevant_features[group])
                            patient_record['diagnosis_treatment_sequence'].append(dict_tmp)
                    
                    if group == 'radiation_oncology' :
                        if not pd.isna(row[get_date_existance(relevant_features, group)]):
                            dict_tmp = self.extract_columns(row, relevant_features[group])
                            patient_record['diagnosis_treatment_sequence'].append(dict_tmp)
                    
                    if group == 'resection_specimen':
                        if not pd.isna(row[get_date_existance(relevant_features, group)]):
                            dict_tmp = self.extract_columns(row, relevant_features[group])
                            patient_record['diagnosis_treatment_sequence'].append(dict_tmp)


                    # Genrow[get_date_existance(relevant_features, group)]eral extraction for other groups
                    # elif row[renamed_features[0]] is not None:
                    #     dict_tmp = self.extract_columns(row, relevant_features[group])
                    #     patient_record[group] = dict_tmp

            records.append(patient_record)
        
        return records


    def episode_restructure_data(self, treatment_sequence: list, timeline: bool = False, follow_up_date=None, patient_static_data=None) -> list:
        """
        Aggregates the treatment sequence into episodes based on treatment dates.
        It uses either "date_field", "date_sarcomaboard" or falls back to "date" to get the treatment date.
        
        If timeline=False: An episode groups treatments that occur within a threshold (e.g., 30 days) of the previous one.
        If timeline=True: Episodes are 6-month periods starting from the first diagnosis/patient record time.
                          If follow_up_date is provided, episodes will extend to cover the follow-up period.
        
        Args:
            treatment_sequence (list): List of treatment dictionaries
            timeline (bool): Whether to use timeline-based (6-month) or threshold-based (30-day) episodes
            follow_up_date: The patient's follow-up date to extend episode coverage
            patient_static_data (dict): Patient's static data containing final status information
        
        Returns a list of episode dictionaries with the following structure:
            {
                "episode_id": int,
                "start_date": ISODate string,
                "end_date": ISODate string,
                "duration_days": int,
                "n_interventions": int,
                "treatments": [dict, ...]
            }
        """
        
        def get_treatment_date(treatment):
            # Collect all valid dates from the treatment
            dates = []
            for key in ['date_field', 'date_sarcomaboard']:
                if key in treatment and treatment[key]:
                    date_val = treatment[key]
                    # If already a datetime, return it
                    if isinstance(date_val, datetime):
                        return date_val
                    # Else try parsing the ISO string (remove trailing "Z" if present)
                    try:
                        return datetime.fromisoformat(date_val.replace("Z", "+00:00"))
                    except Exception:
                        continue
            return None  
        
        def parse_follow_up_date(follow_up_date):
            """Parse follow-up date from various formats"""
            if not follow_up_date:
                return None
            if isinstance(follow_up_date, datetime):
                return follow_up_date
            if isinstance(follow_up_date, str):
                try:
                    return datetime.fromisoformat(follow_up_date.replace("Z", "+00:00"))
                except Exception:
                    try:
                        # Try parsing with standard date parsing logic
                        return self.parse_date(follow_up_date)
                    except Exception:
                        return None
            return None
        
        # Filter out treatments with no valid date and sort by date.
        treatments_with_date = []
        for treatment in treatment_sequence:
            t_date = get_treatment_date(treatment)
            if t_date is not None:
                treatment['_parsed_date'] = t_date
                treatments_with_date.append(treatment)
        treatments_with_date.sort(key=lambda x: x['_parsed_date'])
        
        if not treatments_with_date:
            return []
        
        episodes = []
        
        if timeline:
            # Timeline-based aggregation: 6-month episodes
            
            # Get the earliest date as the baseline
            baseline_date = treatments_with_date[0]['_parsed_date']
            
            # Determine the end date for episodes
            last_treatment_date = treatments_with_date[-1]['_parsed_date']
            parsed_follow_up_date = parse_follow_up_date(follow_up_date)
            
            # Use follow-up date if available and later than last treatment
            if parsed_follow_up_date and parsed_follow_up_date > last_treatment_date:
                end_date = parsed_follow_up_date
            else:
                end_date = last_treatment_date
            
            episode_id = 1
            current_start = baseline_date
            
            while current_start <= end_date:
                current_end = current_start + relativedelta(months=6) - relativedelta(days=1)
                
                # Find treatments in this 6-month period
                episode_treatments = []
                for treatment in treatments_with_date:
                    if current_start <= treatment['_parsed_date'] <= current_end:
                        episode_treatments.append(treatment)
                
                # Create episode
                episode = {
                    "episode_id": episode_id,
                    "start_date": current_start,
                    "end_date": current_end,
                    "duration_days": 180,  # Approximately 6 months
                    "n_interventions": len(episode_treatments),
                    "treatments": episode_treatments
                }
                
                # If no treatments in this period and there are previous episodes,
                # handle gap-filling based on whether this is an intermediary gap or the final episode
                if not episode_treatments and episodes:
                    # Determine if this is the last episode that will be created
                    next_period_start = current_end + relativedelta(days=1)
                    is_last_episode = next_period_start > end_date
                    
                    if is_last_episode:
                        # Last gap-filled episode: ensure compatibility with final status
                        episode["treatments"] = []
                        episode["n_interventions"] = 0
                        
                        # Add minimal final status diagnosis if available
                        if patient_static_data and patient_static_data.get('general', {}).get('status'):
                            final_status = patient_static_data['general']['status']
                            status_treatment = {
                                'section': 'diagnosis',
                                'date_field': current_start.isoformat(),
                                'date_sarcomaboard': current_start.isoformat(),
                                'fields': {
                                    'status': final_status
                                },
                                '_is_final_status': True
                            }
                            
                            # Add death date if patient died
                            if 'date_death' in patient_static_data['general'] and patient_static_data['general']['date_death']:
                                status_treatment['fields']['date_death'] = patient_static_data['general']['date_death']
                            
                            episode["treatments"] = [status_treatment]
                            episode["n_interventions"] = 1
                    
                    else:
                        # Intermediary gap-filled episode: copy diagnosis from most recent episode with treatments
                        # WITHOUT updating status to final status
                        previous_episode = None
                        for prev_ep in reversed(episodes):
                            if prev_ep["treatments"] and any(
                                t.get('section') not in ['diagnosis', 'metastasis'] or 
                                not t.get('_is_copied', False) for t in prev_ep["treatments"]
                            ):
                                previous_episode = prev_ep
                                break
                        
                        if previous_episode:
                            episode["treatments"] = []
                            for treatment in previous_episode["treatments"]:
                                # Only copy diagnosis and metastasis sections
                                if treatment.get('section') in ['diagnosis', 'metastasis']:
                                    copied_treatment = treatment.copy()
                                    # Mark as copied and update dates (but keep original status)
                                    copied_treatment['_is_copied'] = True
                                    if 'date_field' in copied_treatment:
                                        copied_treatment['date_field'] = current_start.isoformat()
                                    if 'date_sarcomaboard' in copied_treatment:
                                        copied_treatment['date_sarcomaboard'] = current_start.isoformat()
                                    copied_treatment['_parsed_date'] = current_start
                                    
                                    # Do NOT update status - keep original diagnosis status
                                    episode["treatments"].append(copied_treatment)
                            
                            episode["n_interventions"] = len(episode["treatments"])
                
                episodes.append(episode)
                episode_id += 1
                current_start = current_end + relativedelta(days=1)
        
        else:
            # Original threshold-based aggregation
            threshold_days = 30
            current_episode = None
            
            for treatment in treatments_with_date:
                t_date = treatment['_parsed_date']
                # Start a new episode if there's no current episode
                # or the gap is larger than threshold_days.
                if (current_episode is None or 
                    (t_date - current_episode['end_date']).days > threshold_days):
                    if current_episode is not None:
                        # Finalize the current episode
                        current_episode['duration_days'] = (current_episode['end_date'] - current_episode['start_date']).days
                        current_episode['n_interventions'] = len(current_episode['treatments'])
                        episodes.append(current_episode)
                    # Start a new episode
                    current_episode = {
                        "episode_id": len(episodes) + 1,
                        "start_date": t_date,
                        "end_date": t_date,
                        "duration_days": 0,
                        "n_interventions": 0,
                        "treatments": []
                    }
                # Add the treatment to the current episode and update end_date.
                current_episode['treatments'].append(treatment)
                if t_date > current_episode['end_date']:
                    current_episode['end_date'] = t_date
            
            # Append the last episode.
            if current_episode is not None:
                current_episode['duration_days'] = (current_episode['end_date'] - current_episode['start_date']).days
                current_episode['n_interventions'] = len(current_episode['treatments'])
                episodes.append(current_episode)
        
        # Remove the temporary parsed date and convert datetimes to ISODate strings.
        for episode in episodes:
            episode['start_date'] = episode['start_date'].isoformat().replace("+00:00", "Z")
            episode['end_date'] = episode['end_date'].isoformat().replace("+00:00", "Z")
            for treatment in episode['treatments']:
                treatment.pop('_parsed_date', None)
        
            episode = self.data_assembler(episode)
        return episodes

    def data_assembler(self, episode):
        # Group treatments by section and merge dictionaries for the same section
        treatments = episode.get("treatments", [])
        episode["surgery"]      = 0
        episode["chemotherapy"] = 0
        episode["radiotherapy"] = 0
        episode["metastasis"]   = 0
        
        # Always initialize diagnosis as an empty array
        episode["diagnosis"] = []
        
        if treatments:
            merged = {}
            for treatment in treatments:
                section = treatment.get("section")
                # assign the treatmens presence
                if section in ["surgery", "chemotherapy", "radiotherapy", "metastasis"]:
                    episode[section] = 1
                if section is not None:
                    if section not in merged:
                        merged[section] = treatment.copy()
                    else:
                        # For diagnosis section, use smarter merging based on timeline
                        if section == 'diagnosis':
                            merged[section] = self._merge_diagnosis_entries(merged[section], treatment, episode)
                        else:
                            # For non-diagnosis sections, use the existing merging logic
                            for key, value in treatment.items():
                                if key in merged[section]:
                                    # If both values are dicts, merge them recursively
                                    if isinstance(merged[section][key], dict) and isinstance(value, dict):
                                        merged_val = merged[section][key].copy()
                                        merged_val.update(value)
                                        merged[section][key] = merged_val
                                    # If both values are lists, concatenate them (last one takes precedence if needed)
                                    elif isinstance(merged[section][key], list) and isinstance(value, list):
                                        merged[section][key] = merged[section][key] + value
                                    else:
                                        # In case of conflict, keep the value from the current treatment (last one)
                                        merged[section][key] = value
                                else:
                                    merged[section][key] = value
            
            
            
            episode = self.split_by_section(episode, list(merged.values()))
        return  episode

    def _merge_diagnosis_entries(self, existing_diagnosis, new_diagnosis, episode):
        """
        Intelligently merges diagnosis entries, prioritizing chronologically appropriate status.
        For timeline-based episodes, we want to use the most appropriate status for the episode period.
        
        Args:
            existing_diagnosis (dict): The existing diagnosis entry
            new_diagnosis (dict): The new diagnosis entry to merge
            episode (dict): The episode context for timing information
            
        Returns:
            dict: The merged diagnosis entry
        """
        def get_diagnosis_date(diagnosis):
            """Extract date from diagnosis entry for comparison"""
            for date_key in ['date_field', 'date_sarcomaboard']:
                if date_key in diagnosis and diagnosis[date_key]:
                    date_val = diagnosis[date_key]
                    if isinstance(date_val, dict) and '$date' in date_val:
                        return date_val['$date']
                    elif isinstance(date_val, str):
                        return date_val
            return None
        
        def parse_episode_date(date_str):
            """Parse episode date string to comparable format"""
            if isinstance(date_str, str):
                return date_str.replace('T00:00:00', '').replace('Z', '')
            return date_str
        
        # Get episode timing
        episode_start = parse_episode_date(episode.get('start_date', ''))
        episode_end = parse_episode_date(episode.get('end_date', ''))
        
        # Get diagnosis dates
        existing_date = get_diagnosis_date(existing_diagnosis)
        new_date = get_diagnosis_date(new_diagnosis)
        
        # If we have dates, prefer the diagnosis that's chronologically more appropriate for this episode
        if existing_date and new_date:
            existing_date_parsed = existing_date.replace('T00:00:00.000Z', '').replace('.000Z', '')
            new_date_parsed = new_date.replace('T00:00:00.000Z', '').replace('.000Z', '')
            
            # For the first episode especially, prefer earlier diagnosis entries over later ones
            # This prevents later status changes (like DOD) from overriding earlier AWD status
            if episode.get('episode_id') == 1:
                # For first episode, prefer the earlier diagnosis
                base_diagnosis = existing_diagnosis if existing_date_parsed <= new_date_parsed else new_diagnosis
                merge_from = new_diagnosis if existing_date_parsed <= new_date_parsed else existing_diagnosis
            else:
                # For later episodes, use the more recent diagnosis
                base_diagnosis = existing_diagnosis if existing_date_parsed >= new_date_parsed else new_diagnosis
                merge_from = new_diagnosis if existing_date_parsed >= new_date_parsed else existing_diagnosis
        else:
            # If no dates available, keep existing as base
            base_diagnosis = existing_diagnosis.copy()
            merge_from = new_diagnosis
        
        # Create merged result starting with the base diagnosis
        merged_result = base_diagnosis.copy()
        
        # Merge fields from the other diagnosis, but preserve critical status information from base
        for key, value in merge_from.items():
            if key in merged_result:
                # Special handling for fields dict - merge carefully
                if key == 'fields' and isinstance(merged_result[key], dict) and isinstance(value, dict):
                    merged_fields = merged_result[key].copy()
                    for field_key, field_value in value.items():
                        # Don't override status from base diagnosis unless base doesn't have it
                        if field_key == 'status' and 'status' in merged_fields:
                            continue  # Keep the base status
                        else:
                            merged_fields[field_key] = field_value
                    merged_result[key] = merged_fields
                # If both values are dicts, merge them recursively
                elif isinstance(merged_result[key], dict) and isinstance(value, dict):
                    merged_val = merged_result[key].copy()
                    merged_val.update(value)
                    merged_result[key] = merged_val
                # If both values are lists, concatenate them
                elif isinstance(merged_result[key], list) and isinstance(value, list):
                    merged_result[key] = merged_result[key] + value
                # For other conflicts, keep the base value (don't override)
            else:
                merged_result[key] = value
        
        return merged_result

    def split_by_section(self, episode, merged_sections):
        """
        Splits merged sections into treatments and diagnosis.
        
        Args:
            episode (dict): The episode dictionary to update.
            merged_sections (iterable): A list of dictionaries representing merged treatment sections.
        
        Returns:
            dict: Updated episode with 'treatments' (as a list) and 'diagnosis' (if found).
        """
        treatments = []
        diagnosis = []
        for item in merged_sections:
            sec = item.get('section')
            if sec in ['surgery', 'chemotherapy', 'radiotherapy']:
                treatments.append(item)

            elif sec in ['diagnosis', 'metastasis']:
                diagnosis.append(item)

        if treatments:
            episode['treatments'] = treatments
        else:
            episode['treatments'] = []
        
        # Only update diagnosis if we found diagnosis items, otherwise keep the initialized empty array
        if diagnosis:
            episode['diagnosis'] = diagnosis
        # If no diagnosis found, episode['diagnosis'] remains as initialized empty array []
        
        return episode

    def  restructure_to_records(self, df: pd.DataFrame, relevant_features: dict, key_column: str, timeline: bool = False) -> list:
        """
        Restructures the data from the dataframe into a list of patient records.

        Parameters:
        df (pd.DataFrame): The input dataframe containing the data.
        relevant_features (dict): A dictionary where keys are group names and values are lists of features.
        key_column (str): The column name to group by, typically a patient identifier.
        timeline (bool): If True, uses 6-month timeline-based episodes; if False, uses 30-day threshold episodes.

        Returns:
        list: A list of dictionaries, each representing a patient record.
        """

        records = []
        
        for group_idx, (patient_id, group_df) in enumerate(df.groupby(key_column), start=1):
            
            patient_record = {"_id": patient_id}
            
            row = group_df.iloc[-1]

            for group_name in relevant_features['static'].keys():
                patient_record[group_name] = self.extract_columns(row, relevant_features['static'][group_name])

            #TODO: adapt 
            # group_df = self._sort_by_column(group_df, column=['patient_id', 'time_relative_sarcomaboard_presentation'], ascending=True)
                        
            patient_record['diagnosis_treatment_sequence'] = []
            for idx, (index, row) in enumerate(group_df.iterrows(), start=1):
                             
                for group, features in relevant_features['dynamic'].items():

                    dict_tmp = self.extract_columns(row, relevant_features['dynamic'][group])
                    if dict_tmp.get('fields', None):
                        if (group == 'surgery' and row.get('surgery_flag', None) == 1) | \
                            (group == 'chemotherapy' and row.get('chemotherapy_flag', None) == 1) | \
                            (group == 'radiotherapy' and row.get('radiation_oncology_flag', None) == 1) | \
                            (group == 'diagnosis') | \
                            (group == 'metastasis' and row.get('metastasis_flag', None) == 1) :

                            dict_tmp['section'] = group
                            patient_record['diagnosis_treatment_sequence'].append(dict_tmp)
            
            # Extract follow-up date for timeline episodes
            follow_up_date = None
            if timeline and patient_record.get('general', {}).get('date_follow_up'):
                follow_up_date = patient_record['general']['date_follow_up']
                    
            patient_record['episodes'] = self.episode_restructure_data(
                patient_record['diagnosis_treatment_sequence'], 
                timeline=timeline, 
                follow_up_date=follow_up_date,
                patient_static_data=patient_record  # Pass the entire patient record for static data access
            )
            records.append(patient_record)
        
        return records
    

    
    def process_restructured_data(self, records):
        for record in records:
            list_docs = list(set(record.keys()) - set(['status_metrics', 'patient_info', 'dates', '_id']))
            record['patient_info']['list_docs'] = list_docs

        return records
    
    def print_sample(self,records, n):
        for i, (key) in enumerate(records):
            if i < n:
                print(f"{key}")
            else:
                break

    def export_sample(self, records, n):
        with open('sample_docs.txt', 'w') as file:
            for i, (key) in enumerate(records):
                if i < n:
                    output = f"{key}\n"
                    print(output.strip())
                    file.write(output)
                else:
                    break


    def import_json(self, file_path):
        """
        This function reads a JSON file and returns the data.
        
        Parameters:
        file_path (str): The path to the JSON file.
        
        Returns:
        dict: The data read from the JSON file.
        """
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            return data
        except FileNotFoundError:
            print(f"The file {file_path} does not exist.")
            return None
        except json.JSONDecodeError:
            print(f"The file {file_path} is not a valid JSON.")
            return None
    
    def export_json(self, records, json_file) :
        with open('buffer.json', 'w') as json_file:
            json.dump(records, json_file, indent=4)

    def parse_date(self, item):
        """
        Parses a date string into a datetime object.
        
        Parameters:
        item (str): The date string to parse.
        
        Returns:
        datetime.datetime: The parsed datetime object, or None if parsing fails.
        """
        if isinstance(item, (int, float)):
            try:
                return pd.to_datetime(item, unit='ms')
            except Exception:
                pass

        if isinstance(item, str):
            for fmt in ('%d.%m.%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y'):
                try:
                    return datetime.strptime(item, fmt)
                except ValueError:
                    continue
                    
        try:
            return pd.to_datetime(item, errors='coerce')
        except Exception:
            return None

    def convert_list_elements(self, data_type: Any, lst: List[Any]) -> List[Any]:
        """
        Converts the elements of a list to the specified data type.
        
        Parameters:
        data_type (Any): The target data type for conversion.
        lst (List[Any]): The list to convert.
        
        Returns:
        List[Any]: The converted list.
        """
        if data_type == 'datetime':
            return [self.parse_date(item) for item in lst]
            # return [datetime.combine(pd.to_datetime(item, errors='coerce', dayfirst=True).date(), datetime.min.time()) for item in lst]
        elif data_type == int:
            converted_list = [pd.to_numeric(item, errors='coerce') for item in lst]
            converted_list = [int(item) if pd.notnull(item) else None for item in converted_list]
            return converted_list
        elif data_type == float:
            return [pd.to_numeric(item, errors='coerce') for item in lst]
        
            
        else:
            return lst

    def covert_single_element_column(self, data_type: Any, column: pd.Series) -> pd.Series:
        """
        Converts the elements of a pandas Series column to the specified data type.
        
        Parameters:
        data_type (Any): The target data type for conversion.
        column (pd.Series): The pandas Series column to convert.
        
        Returns:
        pd.Series: The converted pandas Series column.
        """
        if data_type == 'datetime':
            return column.apply(lambda x: self.parse_date(x) if pd.notnull(x) else None)
            # return column.apply(lambda x: datetime.combine(pd.to_datetime(x, errors='coerce', dayfirst=True).date(), datetime.min.time()) if pd.notnull(pd.to_datetime(x, errors='coerce', dayfirst=True)) else None)
        elif data_type == int:
            numeric_column = pd.to_numeric(column, errors='coerce')
            numeric_column = numeric_column.fillna(-1).astype(int)  # .astype(pd.Int64Dtype())
            return numeric_column.replace(-1, None)
        elif data_type == float:
            return pd.to_numeric(column, errors='coerce')
        else:
            return column

    def convert_single_element(self, data_type: Any, element: Any) -> Union[int, float, Any]:
        """
        Converts a single element to the specified data type.
        
        Parameters:
        data_type (Any): The target data type for conversion.
        element (Any): The element to convert.
        
        Returns:
        Union[int, float, Any]: The converted element.
        """
        if data_type == 'datetime':
            return self.parse_date(element) if pd.notnull(element) else None
        elif data_type == int:
            numeric_element = pd.to_numeric(element, errors='coerce')
            return int(numeric_element) if pd.notnull(numeric_element) else None
        elif data_type == float:
            return pd.to_numeric(element, errors='coerce')
        elif isinstance(element, str):
            if re.match(r'^\s*$', element):
                return None
            else:
                return element
        else:
            return element

    def convert_column(self, column: pd.Series, data_type: Any) -> pd.Series:
        """
        Converts the elements of a pandas Series column to the specified data type.
        
        Parameters:
        column (pd.Series): The pandas Series column to convert.
        data_type (Any): The target data type for conversion.
        
        Returns:
        pd.Series: The converted pandas Series column.
        """
        if column.apply(lambda x: isinstance(x, list)).any():
            # If any element in the column is a list, convert each element in the list
            return column.apply(lambda x: self.convert_list_elements(data_type, x) if isinstance(x, list) else self.convert_single_element(data_type, x))
        else:
            # If no elements are lists, convert each element directly
            return column.apply(lambda x: self.convert_single_element(data_type, x))

    def process_list_column(self, column: pd.Series) -> pd.Series:
        """
        Processes a pandas Series column by removing duplicates from lists within the column.
        
        Parameters:
        column (pd.Series): The pandas Series column to process.
        
        Returns:
        pd.Series: The processed pandas Series column with duplicates removed from lists.
        """
        if column.apply(lambda x: isinstance(x, list)).any():
            # If any element in the column is a list, remove duplicates from each list
            column = column.apply(lambda x: list(set(x)) if isinstance(x, list) else x)
        return column      

    def transform_nested_values(self, df):
        """
        Transforms nested values in cells by splitting on common separators (;, /, |).
        If a cell contains multiple values separated by these delimiters, converts them into a list.
        
        Args:
            df (pd.DataFrame): The input DataFrame to be transformed.
            features_mapping (dict): A dictionary mapping features (optional, can be used to target specific columns).
        
        Returns:
            pd.DataFrame: The DataFrame with nested values split into lists.
        """
        def split_nested_values(value):
            """Helper function to split a value by separators and return as list or original value."""
            if pd.isna(value):
                return value
            
            if isinstance(value, list):
                # If already a list, recursively process each element
                return [split_nested_values(item) for item in value]
            
            if isinstance(value, str):
                # Check if value contains any of the separators
                if ';' in value or '/' in value or '|' in value:
                    # Split by any of the separators and clean up whitespace
                    # Use regex to split by multiple separators
                    import re
                    parts = re.split(r'[;/|]', value)
                    # Strip whitespace from each part and remove empty strings
                    cleaned_parts = [part.strip() for part in parts if part.strip()]
                    # Return as list if we have multiple parts, otherwise return single value
                    return cleaned_parts if len(cleaned_parts) > 1 else (cleaned_parts[0] if cleaned_parts else value)
            
            return value
        
        # Apply the transformation to all cells in the dataframe
        for column in df.columns:
            df[column] = df[column].apply(split_nested_values)
        
        return df

    # Define the main transformation function
    def transform_dataframe(self, df: pd.DataFrame, data_format_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts columns in the DataFrame to the appropriate data formats as specified in data_format_dict.

        Args:
            df (pd.DataFrame): The input DataFrame to be transformed.
            data_format_dict (Dict[str, Any]): A dictionary where keys are column names and values are the target data types.

        Returns:
            pd.DataFrame: The DataFrame with columns converted to the specified data formats.
        """
        for column, data_type in data_format_dict.items():
            if column in df.columns:

                df[column] = self.convert_column(df[column], data_type)
                df[column] = self.process_list_column(df[column])
                df[column] = df[column].astype('object')
        
        return df

class FeatureExtractor:
    """
    This class is responsible for extracting relevant features from a DataFrame.
    """

    def __init__(self, df, relevant_features, data_type_mapping = None):
        self.df = df
        self.relevant_features = relevant_features
        self.data_type_mapping = data_type_mapping

    def read_excel(self, file_path):
        return pd.read_excel(file_path, engine='openpyxl')

    def merge_dfs(self, df_merge, key_column, columns_to_merge):
        """
        Merges multiple DataFrames into a single DataFrame based on a key column.

        Args:
            key_column (str): The key column to merge the DataFrames on.
            columns_to_merge (List[pd.DataFrame]): A list of DataFrames to merge.

        Returns:
            pd.DataFrame: The merged DataFrame.
        """
        
        

        df_merge = df_merge[[key_column] + columns_to_merge]
        self.df = pd.merge(self.df, df_merge, left_on='patient_id', right_on=key_column, how='left')
        
    def initial_metastasis_feature(self):
        df_metastasis_luks =self.read_excel('ETL/data/fulltable_luks.xlsx')
        df_metastasis_mets = self.read_excel('ETL/data/fulltable_mets.xlsx')

        self.merge_dfs(df_metastasis_luks, 'pid', ['metastasis'])
        self.df['metastasis_initial_presentation_exists_original'] = self.df['metastasis_initial_presentation_exists']
        for index, row in self.df.iterrows():
            if pd.isna(row['metastasis_initial_presentation_exists']) and row['metastasis'] == 0:
                self.df.at[index, 'metastasis_initial_presentation_exists'] = row['metastasis']

            
        for index, row in self.df.iterrows():
            if pd.isna(row['metastasis_initial_presentation_exists']) and row['metastasis'] == 1:
                pid = row['pid']
                mets_rows = df_metastasis_mets[(df_metastasis_mets['pid'] == pid) & 
                                            (df_metastasis_mets['event_type'] == 'Initiale Sarkoma Diagnose')]
                if not mets_rows['metastasis_location'].isna().all():
                    self.df.at[index, 'metastasis_initial_presentation_exists'] = 1
        # self.df['metastasis_initial_presentation_exists_original'] =self.df['metastasis_initial_presentation_exists_original'].astype('int64')
        
        self.df['metastasis_initial_presentation_exists_original'] = self.df['metastasis_initial_presentation_exists_original'].astype('float64')
        self.df['metastasis_initial_presentation_exists_original'] = self.df['metastasis_initial_presentation_exists_original'].round()
        self.relevant_features['initial_diagnosis']['metastasis_initial_presentation_exists_original'] = 'metastasis_initial_presentation_exists_original'
        return self

    def whoops_feature(self):
        """
        Adds a 'whoops' feature to the DataFrame if 'date_of_whoops_surgery' column exists.
        The 'whoops' feature is set to 1 if 'date_of_whoops_surgery' is not null, otherwise it is set to None.
        The 'whoops' feature is added to the 'relevant_features' list.

        Returns:
            self: The instance of the data_transfomer class.
        """
        if 'date_of_whoops_surgery' in self.df.columns:
            whoops_series = self.df['date_of_whoops_surgery'].notnull().apply(lambda x: 1 if x else None)
            self.df = pd.concat([self.df, whoops_series.rename('whoops')], axis=1)
            self.relevant_features['operations_n'].extend(['whoops'])
        return self

    def convert_column(self, column: pd.Series, data_type: Any) -> pd.Series:
        """
        Converts the elements of a pandas Series column to the specified data type.
        
        Parameters:
        column (pd.Series): The pandas Series column to convert.
        data_type (Any): The target data type for conversion.
        
        Returns:
        pd.Series: The converted pandas Series column.
        """
        if column.apply(lambda x: isinstance(x, list)).any():
            # If any element in the column is a list, convert each element in the list
            return column.apply(lambda x: self.convert_list_elements(data_type, x) if isinstance(x, list) else self.convert_single_element(data_type, x))
        else:
            # If no elements are lists, convert each element directly
            return column.apply(lambda x: self.convert_single_element(data_type, x))

    def process_list_column(self, column: pd.Series) -> pd.Series:
        """
        Processes a pandas Series column by removing duplicates from lists within the column.
        
        Parameters:
        column (pd.Series): The pandas Series column to process.
        
        Returns:
        pd.Series: The processed pandas Series column with duplicates removed from lists.
        """
        if column.apply(lambda x: isinstance(x, list)).any():
            # If any element in the column is a list, remove duplicates from each list
            column = column.apply(lambda x: list(set(x)) if isinstance(x, list) else x)
        return column  
    
    def convert_list_elements(self, data_type: Any, lst: List[Any]) -> List[Any]:
        """
        Converts the elements of a list to the specified data type.
        
        Parameters:
        data_type (Any): The target data type for conversion.
        lst (List[Any]): The list to convert.
        
        Returns:
        List[Any]: The converted list.
        """
        if data_type == 'datetime':
            return [self.parse_date(item) for item in lst]
            # return [datetime.combine(pd.to_datetime(item, errors='coerce', dayfirst=True).date(), datetime.min.time()) for item in lst]
        elif data_type == int:
            converted_list = [pd.to_numeric(item, errors='coerce') for item in lst]
            converted_list = [int(item) if pd.notnull(item) else None for item in converted_list]
            return converted_list
        elif data_type == float:
            return [pd.to_numeric(item, errors='coerce') for item in lst]
        
            
        else:
            return lst

    def covert_single_element_column(self, data_type: Any, column: pd.Series) -> pd.Series:
        """
        Converts the elements of a pandas Series column to the specified data type.
        
        Parameters:
        data_type (Any): The target data type for conversion.
        column (pd.Series): The pandas Series column to convert.
        
        Returns:
        pd.Series: The converted pandas Series column.
        """
        if data_type == 'datetime':
            return column.apply(lambda x: self.parse_date(x) if pd.notnull(x) else None)
            # return column.apply(lambda x: datetime.combine(pd.to_datetime(x, errors='coerce', dayfirst=True).date(), datetime.min.time()) if pd.notnull(pd.to_datetime(x, errors='coerce', dayfirst=True)) else None)
        elif data_type == int:
            numeric_column = pd.to_numeric(column, errors='coerce')
            numeric_column = numeric_column.fillna(-1).astype(int)  # .astype(pd.Int64Dtype())
            return numeric_column.replace(-1, None)
        elif data_type == float:
            return pd.to_numeric(column, errors='coerce')
        else:
            return column
        
    def convert_single_element(self, data_type: Any, element: Any) -> Union[int, float, Any]:
        """
        Converts a single element to the specified data type.
        
        Parameters:
        data_type (Any): The target data type for conversion.
        element (Any): The element to convert.
        
        Returns:
        Union[int, float, Any]: The converted element.
        """
        if data_type == 'datetime':
            return self.parse_date(element) if pd.notnull(element) else None
        elif data_type == int:
            numeric_element = pd.to_numeric(element, errors='coerce')
            return int(numeric_element) if pd.notnull(numeric_element) else None
        elif data_type == float:
            return pd.to_numeric(element, errors='coerce')
        elif isinstance(element, str):
            if re.match(r'^\s*$', element):
                return None
            else:
                return element
        else:
            return element

    def get_relevant_features(self):
        """
        Returns the relevant features.

        Returns:
            list: The list of relevant features.
        """
        return self.relevant_features

    def get_missing_data_index(self):
        missing_data_index = self.df.groupby('patient_id').apply(lambda x: round(x.isnull().mean().mean(), 2)).reset_index(name='missing_data_index')
        self.df = pd.merge(self.df, missing_data_index, on='patient_id', how='left')
        self.relevant_features['status_metrics']['missing_data_index'] = 'missing_data_index'
        return self


    def get_charlson_index(self):
        """
        Computes the Charlson Comorbidity Index (CCI) for each patient and adds it to the DataFrame.
        
        Returns:
            self: Updates the DataFrame with the Charlson Index.
        """

        def solid_tumor_localized(row):

            if 'diagnosis_malignant_benign' in row:
                if row['diagnosis_malignant_benign'] == 3:
                    return 2
                else:
                    return 0
            
            return 0
        
        def solid_tumor_metastasis(row):
            if 'diagnosis_malignant_benign' in row:
                if row['presence_of_metastasis'] == 1:
                    return 6
                else:
                    return 0
            return 0
        
        def age_points(age):
            if age >= 50 and age <= 59:
                return 1
            elif age >= 60 and age <= 69:
                return 2
            elif age >= 70 and age <= 79:
                return 3
            elif age >= 80:
                return 4
            else:
                return 0

        # Compute the Charlson Index for each patient
        self.df['charlson_index'] = self.df.apply(
            lambda row: (
                # Add age points
                age_points(row['age_at_admission']) +
                solid_tumor_localized(row) + 
                solid_tumor_metastasis(row)
            ),
            axis=1
        )

        self.df['charlson_index'] = self.df['charlson_index'].astype('float64')
        self.df['charlson_index'] = self.df['charlson_index'].round()
        self.relevant_features['status_metrics']['charlson_index'] = 'charlson_index'
        
        return self

    def get_timeframe_entry_last_update(self):
            self.df['date_of_last_patient_contact'] = pd.to_datetime(self.df['date_of_last_patient_contact'], errors='coerce')
            self.df['entry_date'] = pd.to_datetime(self.df['entry_date'], errors='coerce')
            time_diff = self.df.groupby('patient_id').apply(
                            lambda x: (x['date_of_last_patient_contact'].max(skipna=True) - x['entry_date'].min(skipna=True)).days
                        ).reset_index(name='timeframe_entry_last_update')
            self.df = pd.merge(self.df, time_diff, on='patient_id', how='left')
            self.relevant_features['dates'].extend(['timeframe_entry_last_update'])
            return self

    def process(self):
        """
        Process the data and return the transformed DataFrame.

        Returns:
            pandas.DataFrame: The transformed DataFrame.
        """
        return (self.whoops_feature()
                .initial_metastasis_feature()
                .get_charlson_index()
                .get_missing_data_index()
                .get_timeframe_entry_last_update()
                .df
                )
    

    def mock_process(self):
        return (self.df)
    
    
    def get_age(self):
        
        self.df['age']  = datetime.today().year - self.df['birth_date'] 

        self.relevant_features['static']['general']['age'] = 'age'
        
        self.data_type_mapping['age'] = int
        return self

    def get_single_sarcomaboard_date(self):
        self.df['date_sarcomaboard'] = self.df['date_sarcomaboard'].apply(lambda x: x[0] if isinstance(x, list) and x else x)
        return self

    def get_metastasis_after_first_treatment(self):
        metastasis_after_first_treatment = self.df.groupby('patient_id')['follow_up'].apply(
            lambda group: 1 if (group == 1).any() else 0
        ).reset_index(name='metastasis_after_first_treatment')
        self.df = pd.merge(self.df, metastasis_after_first_treatment, on='patient_id', how='left')
        self.relevant_features['static']['general']['metastasis_after_first_treatment'] = 'metastasis_after_first_treatment'
        return self
    
    def get_days_histological_diagnosis(self):
        self.df['date_first_patientcontact'] = pd.to_datetime(self.df['date_first_patientcontact'], errors='coerce')
        self.df['date_of_diagnosis'] = pd.to_datetime(self.df['date_of_diagnosis'], errors='coerce')
        diff = (self.df['date_of_diagnosis'] - self.df['date_first_patientcontact']).dt.days
        self.df['days_histological_diagnosis'] = diff.where(diff > 0, None)
    
        self.relevant_features['static']['general']['days_histological_diagnosis'] = 'days_histological_diagnosis'
        self.data_type_mapping['days_histological_diagnosis'] = int
        return self
    
    # def get_cleaning_mitotic(self):
    #     self.df['resection_mitotic'] = self.df['resection_mitotic'].apply(lambda x: x if x in [1, 2, 3] else None)
    #     return self
    
    # def get_cleaning_necrosis(self):
    #     self.df['resection_necrosis'] = self.df['resection_necrosis'].apply(lambda x: x if x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] else None)
    #     return self

    def get_clone(self):
        # self.df['histological_diagnosis_clone'] = self.df['histological_diagnosis'].copy()
        self.relevant_features['static']['tumor_characteristics']['histological_diagnosis_clone'] = 'histological_diagnosis_clone'
        self.data_type_mapping['histological_diagnosis_clone'] = str
        return self
    
    def get_general_biopsy_adjuvant(self):
        """
        Extracts general biopsy and adjuvant treatment information from the DataFrame.

        Returns:
            self: Updates the DataFrame with the extracted features.
        """
        self.df = self.df.groupby('patient_id').apply(
            lambda group: group.assign(
            biopsy_neoadjuvant_general=group['biopsy_neoadjuvant']
                .dropna()
                .apply(lambda x: x if not isinstance(x, list) else list(set(x)))
                .mode().iloc[0] if not group['biopsy_neoadjuvant'].dropna().empty else None
            )
        ).reset_index(drop=True)
        self.relevant_features['static']['general']['biopsy_neoadjuvant_general'] = 'biopsy_neoadjuvant_general'
        self.data_type_mapping['biopsy_neoadjuvant_general'] = int
        return self
    
    def get_number_surgery_reoperations(self):
        self.df['number_surgery_reoperations'] = self.df['number_surgeries'] -1 
        self.relevant_features['dynamic']['surgery']['fields']['number_surgery_reoperations'] = 'number_surgery_reoperations'
        self.data_type_mapping['number_surgery_reoperations'] = int
        return self
    
    def get_match_chemo_indication(self):
        #TODO: this needs to be improved
        mapping = {
            1: "curative intent: neo-adjuvant",
            2: "curative intent: adjuvant",
            3: "palliative intent: treatment pressure",
            4: "palliative intent: maintenance"        }
        self.df['chemo_indication'] = self.df['chemo_indication'].apply(
            lambda x: [mapping.get(item, item) for item in x] if isinstance(x, list) else mapping.get(x, x)
        )

        return self

    def get_duration_chemotherapy(self):

        self.df['chemo_start'] = self.df['chemo_start'].apply(
            lambda x: [pd.to_datetime(date.strip(), errors='coerce') for date in x.split('|')] if isinstance(x, str) else x
        )
        self.df['chemo_end'] = self.df['chemo_end'].apply(
            lambda x: [pd.to_datetime(date.strip(), errors='coerce') for date in x.split('|')] if isinstance(x, str) else x
        )

        def calculate_chemo_duration(row):
            # Get the first date from chemo_start list
            def parse_date(val):
                if isinstance(val, pd.Timestamp):
                    return val
                try:
                    parsed = pd.to_datetime(val, errors='coerce')
                    if pd.notnull(parsed):
                        return parsed
                except Exception:
                    return None
                return None

            # If chemo_start is a list, sum the differences pairwise
            if isinstance(row['chemo_start'], list):
                # If chemo_end is also a list, process each pair
                if isinstance(row['chemo_end'], list):
                    total_days = 0
                    valid_durations = 0
                    for start_val, end_val in zip(row['chemo_start'], row['chemo_end']):
                        start_date = parse_date(start_val)
                        end_date = parse_date(end_val)
                        if start_date is None or end_date is None:
                            continue
                        days_diff = (end_date - start_date).days
                        # Only add positive durations and count valid ones
                        if days_diff > 0:
                            total_days += days_diff
                            valid_durations += 1
                    # Return total only if we have valid positive durations
                    return total_days if valid_durations > 0 and total_days > 0 else None
                else:
                    start_date = parse_date(row['chemo_start'][0])
                    end_date = parse_date(row['chemo_end'])
                    if start_date is None or end_date is None:
                        return None
                    days_diff = (end_date - start_date).days
                    # Explicitly check for positive duration only
                    return days_diff if days_diff > 0 else None
            else:
                # chemo_start is a single date
                start_date = parse_date(row['chemo_start'])
                end_date = parse_date(row['chemo_end'])
                if start_date is None or end_date is None:
                    return None
                days_diff = (end_date - start_date).days
                # Explicitly check for positive duration only
                return days_diff if days_diff > 0 else None

        self.df['duration_chemotherapy'] = self.df.apply(calculate_chemo_duration, axis=1)
        self.relevant_features['dynamic']['chemotherapy']['fields']['duration_chemotherapy'] = 'duration_chemotherapy'
        self.data_type_mapping['duration_chemotherapy'] = int
        return self
    
    
    def get_radiation_oncology_indication(self):
        mapping = {
            1: "preoperative",
            2: "postoperative",
            3: "intraoperativ",
            4: "stereotactic",
            5: "palliativ"
        }

        def map_value(value):
            try:
                num = int(value)
                return mapping.get(num, None)
            except (ValueError, TypeError):
                return None

        def map_indication(val):
            if isinstance(val, list):
                return [map_value(item) for item in val]
            else:
                return map_value(val)

        if 'redonc_indication' in self.df.columns:
            self.df['redonc_indication'] = self.df['redonc_indication'].apply(map_indication)
            self.data_type_mapping['redonc_indication'] = str
        return self


    def get_substance_chemotherapy_mapping(self):
        
        mapping = {
            1: "Cyclophosphamid",
            2: "Dacarbazin",
            3: "Docetaxel",
            4: "Doxorubicin",
            5: "Ecteinascidin 743",
            6: "Epirubicin",
            7: "Eribulin",
            8: "Gemzitabin",
            9: "Ifosfamid",
            10: "Letrozol",
            11: "Olaparib",
            12: "Paclitaxel",
            13: "Temozolomid",
            14: "Trofosfamid",
            15: "Vincristin",
            22: "Cisplatin",
            23: "Methotrexat",
            24: "Carboplatin",
            25: "Tamoxifen",
            26: "Etoposid",
            27: "Actinomycin D",
            28: "Everolimus",
            29: "Paraplatin",
            30: "Brigimadlin",
            31: "Topotecan",
            32: "Idarubicin",
            33: "Irinotecan",
            34: "MESNA",
            35: "Gestagen",
            36: "Lu-177 PSMA",
            37: "Idasanutlin",
            39: "Cabozantinib",
            41: "Regorafenib",
            44: "Palbociclib",
            45: "Crizotinib",
            16: "Axitinib",
            17: "Bevacizumab",
            18: "Imatinib",
            19: "Olaratumab",
            20: "Pazopanib",
            21: "Sunitinib",
            38: "Pembrolizumab",
            40: "Nivolumab",
            42: "Atezolizumab",
            43: "Ipilimumab"
        }
        
        def transform_item(val):
            try:
                int_val = int(val)
            except (ValueError, TypeError):
                int_val = None
            return mapping.get(int_val, None)

        def split_substances(val):
            if isinstance(val, list):
                new_list = []
                for item in val:
                    if isinstance(item, str) and ',' in item:
                        new_list.extend([entry.strip() for entry in item.split(',')])
                    else:
                        new_list.append(item)
                return new_list
            return val

        # First, split substances if necessary
        self.df['chemo_substance'] = self.df['chemo_substance'].apply(split_substances)

        def map_to_dict(val):
            # Ensure we work with a list
            if not isinstance(val, list):
                val = [val]
            # Transform items using the mapping
            substances = [transform_item(item) for item in val]
            substances = [s for s in substances if s is not None]
            # Create a dictionary for every treatment in the mapping
            all_treatments = set(mapping.values())
            result = {treatment: 0 for treatment in all_treatments}
            for substance in substances:
                result[substance] = 1
            return result

        self.df['chemo_substance'] = self.df['chemo_substance'].apply(map_to_dict)
        return self

    def get_mapping_chemo_response(self):
        mapping = {
            1: "toxicity",
            2: "Progressive disease",
            3: "planned",
            4: "Patient wish",
            5: "death",
            6: "unfinished/open"
        }
        
        def convert_item(item):
            try:
                return mapping.get(int(item), None) if item is not None else None
            except ValueError:
                return None
        self.df['chemo_discontinuation'] = self.df['chemo_discontinuation'].apply(
            lambda x: [convert_item(item) for item in x] if isinstance(x, list) else convert_item(x)
        )
        def binary_conversion(val):
            if isinstance(val, list):
                if "planned" in val:
                    return 1
                elif any(item in val for item in ["toxicity", "Progressive disease", "patient wish", "death"]):
                    return 0
                else:
                    return None
            else:
                if val == "planned":
                    return 1
                elif val in ["toxicity", "Progressive disease", "patient wish", "death"]:
                    return 0
                else:
                    return None

        self.df['chemo_response_binary'] = self.df['chemo_discontinuation'].apply(binary_conversion)
        self.relevant_features['dynamic']['chemotherapy']['fields']['chemo_response_binary'] = 'chemo_response_binary'
        self.data_type_mapping['chemo_response_binary'] = int
        return self

    def get_chemo_response(self):
        """
        Extracts the chemotherapy response from the DataFrame and adds it to the relevant features.

        Returns:
            self: The instance of the FeatureExtractor class.
        """
        response_mapping = {
            1: "complete remission",
            2: "partial remission",
            3: "stable remission",
            4: "Progressive disease",
            "1": "complete remission",
            "2": "partial remission",
            "3": "stable remission",
            "4": "Progressive disease"
        }
        
            # For string values, split by comma then map each stripped element;
            # For list values, map each element directly.
        self.df["chemo_response"] = self.df["chemo_response"].apply(
        lambda x: (
            [response_mapping.get(int(item.strip()), item.strip()) for item in x.split(",")]
            if isinstance(x, str)
            else ([response_mapping.get(item, item) for item in x] if isinstance(x, list) else x)
        )
        )

        return self
    
    def get_therapy_flag(self):
        #surgery_flag
        self.df['surgery_flag'] = self.df.apply(
            lambda row: 1 if pd.notnull(row['type_surgery']) else 0,
            axis=1
        )
        self.relevant_features['dynamic']['surgery']['surgery_flag'] = 'surgery_flag'
        self.data_type_mapping['surgery_flag'] = int
        
        #radiation_oncology_flag
        self.df['radiation_oncology_flag'] = self.df.apply(
            lambda row: 1 if pd.notnull(row['radiotherapy_indication']) and row['radiotherapy_indication'] != 'no radiotherapy' else 0,
            axis=1
        )
        self.relevant_features['dynamic']['radiotherapy']['radiation_oncology_flag'] = 'radiation_oncology_flag'
        self.data_type_mapping['radiation_oncology_flag'] = int
        
        #chemotherapy_flag
        self.df['chemotherapy_flag'] = self.df.apply(lambda row: 1 if pd.notnull(row['chemo_reason']) else 0, axis=1)
        self.relevant_features['dynamic']['chemotherapy']['chemotherapy_flag'] = 'chemotherapy_flag'
        self.data_type_mapping['chemotherapy_flag'] = int

        self.df['metastasis_flag'] = self.df.apply(
            lambda row: 0 if (pd.isna(row['presence_metastasis']) or row['presence_metastasis'] == 0.0) else 1,
            axis=1
        )
        self.relevant_features['dynamic']['metastasis']['metastasis_flag'] = 'metastasis_flag'
        self.data_type_mapping['metastasis_flag'] = int

        return self


    def get_final_status(self):
        """
        Adjusts the general status for each patient group: if any row has DOD status,
        all rows except the last one are changed to AWD, and the last row remains DOD.
        
        Returns:
            self: Updates the DataFrame with the adjusted status.
        """
        def adjust_status_group(group):
            # Check if any row in the group has DOD status
            if (group['status'] == 'DOD').any():
                # Set all rows to AWD except the last one
                group['status'] = 'AWD'
                # Set the last row to DOD
                group.iloc[-1, group.columns.get_loc('status')] = 'DOD'
            return group
        
        # Apply the adjustment to each patient group
        self.df = self.df.groupby('patient_id').apply(adjust_status_group).reset_index(drop=True)
        self.relevant_features['dynamic']['diagnosis']['fields']['status'] = 'status'
        self.data_type_mapping['status'] = str
        return self

    def get_initial_size(self):
        """
        Adjusts the general status for each patient group: if any row has DOD status,
        all rows except the last one are changed to AWD, and the last row remains DOD.
        
        Returns:
            self: Updates the DataFrame with the adjusted status.
        """
        
        # Apply the adjustment to each patient group
        self.df['initial_size'] = (self.df['initial_size_a']**2 + self.df['initial_size_b']**2 + self.df['initial_size_c']**2)**0.5
        self.relevant_features['dynamic']['diagnosis']['fields']['initial_size'] = 'initial_size'
        self.data_type_mapping['initial_size'] = float
        return self


    def process_mc(self):
        return (self.get_age()
                .get_single_sarcomaboard_date()
                .get_general_features()
                .get_tumor_characteristics()
                .get_static_treatments()
                .get_days_histological_diagnosis()
                .get_general_biopsy_adjuvant()
                # .get_cleaning_mitotic()
                .get_number_surgery_reoperations()
                .get_match_chemo_indication()
                .get_duration_chemotherapy()
                .get_substance_chemotherapy_mapping()
                .get_mapping_chemo_response()
                .get_chemo_response()
                .get_radiation_oncology_indication()
                .get_therapy_flag()
                .get_final_status()
                .get_initial_size()
                .df)
    
    def post_process_mc(self):
        return (self.get_metastasis_after_first_treatment()
                .df)
    
    def process_frozen(self):
        return (self
                .df)
    
    def post_process_frozen(self):
        return (self
                .df)


    
    def assign_train_test_status(self, test_size=0.2, random_state=42):
        """
        Assigns a new integer status to indicate if the patient is in the train or test set based on the distribution of the missing data index.

        Args:
            test_size (float or int): The proportion or absolute number of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before applying the split.

        Returns:
            self: Updates the DataFrame with the train/test status.
        """

        # Bin the missing_data_index values into categories
        self.df['missing_data_index_binned'] = pd.qcut(self.df['missing_data_index'], q=4, labels=False, duplicates='drop')

        # Group by patient_id and aggregate missing_data_index_binned
        grouped_df = self.df.groupby('patient_id')['missing_data_index_binned'].first().reset_index()

        # Determine the number of unique classes
        n_classes = grouped_df['missing_data_index_binned'].nunique()

        # Ensure test size is sufficient to have at least one sample per class
        min_test_size = n_classes
        
        # If test_size is a float, convert to an absolute number of samples
        if isinstance(test_size, float):
            adjusted_test_size = max(ceil(test_size * len(grouped_df)), min_test_size)
        else:
            # If test_size is an int, ensure it's large enough but less than the total number of samples
            adjusted_test_size = max(test_size, min_test_size)

        # Ensure the test size does not exceed the number of available samples
        adjusted_test_size = min(adjusted_test_size, len(grouped_df) - 1)

        # Perform train_test_split with the adjusted test size
        train_ids, test_ids = train_test_split(
            grouped_df['patient_id'],
            test_size=adjusted_test_size,
            random_state=random_state,
            stratify=grouped_df['missing_data_index_binned']
        )
        train_ids_set = set(train_ids)

        # Assign train/test status
        self.df['train_test_status'] = self.df['patient_id'].apply(lambda x: 0 if x in train_ids_set else 1)

        # Check for null values in the 'train_test_status' column
        if self.df['train_test_status'].isnull().any():
            raise ValueError("Null values found in 'train_test_status' column. Please handle them before proceeding.")

        # Update relevant features
        self.df['train_test_status'] = self.df['train_test_status'].astype('float64')
        self.df['train_test_status'] = self.df['train_test_status'].round()
        self.relevant_features['status_metrics']['train_test_status']= 'train_test_status'

        return self

    def first_treatment_doc(self):
                
        pass

    
    def get_first_therapy(self):
        if 'date_of_surgery_dd_mm_yyyy' in self.df.columns:
            
            sorted_df = self.df.sort_values(by=['patient_id', 'date_of_surgery_dd_mm_yyyy'])
            
            first_treatment = sorted_df.drop_duplicates(subset='patient_id', keep='first')[['patient_id', 'therapy_type']]
            first_treatment = first_treatment.rename(columns={'therapy_type': 'first_treatment'})

            def convert_single_element_list(value):
                if isinstance(value, list) and len(value) == 1:
                    return int(value[0])
                return value
            
            self.df = self.df.merge(first_treatment, on='patient_id', how='left')
            self.df['first_treatment'] = self.df['first_treatment'].apply(convert_single_element_list)


            self.relevant_features['patient_info']['first_treatment'] ='first_treatment'
        return self

    def get_first_diagnosis(self):
        if 'diagnosis_malignant_benign' in self.df.columns:
            
            sorted_df = self.df.sort_values(by=['patient_id', 'date_of_sarcomaboard_presentation'])
            
            first_diagnosis = sorted_df.drop_duplicates(subset='patient_id', keep='first')[['patient_id', 'diagnosis_malignant_benign']]
            first_diagnosis = first_diagnosis.rename(columns={'diagnosis_malignant_benign': 'first_diagnosis_malignant_benign' })


            
            self.df = self.df.merge(first_diagnosis, on='patient_id', how='left')
            
            self.df['first_diagnosis_malignant_benign'] = self.df['first_diagnosis_malignant_benign'].astype('float64')
            self.df['first_diagnosis_malignant_benign'] = self.df['first_diagnosis_malignant_benign'].round()
            self.relevant_features['initial_diagnosis']['first_diagnosis_malignant_benign']= 'first_diagnosis_malignant_benign'
        return self

    def get_first_anatomic_region_code_grouping(self):
        if 'anatomic_region_code_grouping' in self.df.columns:
            
            sorted_df = self.df.sort_values(by=['patient_id', 'date_of_sarcomaboard_presentation'])
            
            first_diagnosis = sorted_df.drop_duplicates(subset='patient_id', keep='first')[['patient_id', 'anatomic_region_code_grouping']]
            first_diagnosis = first_diagnosis.rename(columns={'anatomic_region_code_grouping': 'first_anatomic_region_code_grouping' })


            
            self.df = self.df.merge(first_diagnosis, on='patient_id', how='left')
            
            
            self.relevant_features['initial_diagnosis']['first_anatomic_region_code_grouping']= 'first_anatomic_region_code_grouping'
        return self


    def get_discontinuation_chemotherapy(self):
        """
        Adds a column 'discontinuation_chemotherapy' to the DataFrame.
        The column will have a value of 1 if 'number_of_cycles_planned' is equal to 'number_of_cycles_executed', otherwise 0.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame containing the columns 'number_of_cycles_planned' and 'number_of_cycles_executed'.
        
        Returns:
        pd.DataFrame: The DataFrame with the new column 'discontinuation_chemotherapy'.
        """
        self.df['discontinuation_chemotherapy'] = self.df.apply(
            lambda row: None if pd.isna(row['number_of_cycles_planned']) else (1 if row['number_of_cycles_planned'] == row['number_of_cycles_executed'] else 0),
            axis=1
        )
        self.relevant_features['operations_n'].extend(['discontinuation_chemotherapy'])
        return self

    def get_duration_radiotherapy(self):
        # Convert columns to datetime
        self.df['start_of_radonc'] = pd.to_datetime(self.df['start_of_radonc'], errors='coerce')
        self.df['end_of_radonc'] = pd.to_datetime(self.df['end_of_radonc'], errors='coerce')

        # Calculate duration for each row
        self.df['duration_radiotherapy'] = (self.df['end_of_radonc'] - self.df['start_of_radonc']).dt.days

        # Extend relevant features
        self.relevant_features['operations_n'].extend(['duration_radiotherapy'])

        return self

    def flatten_list(self, column):
        # Ensure each element is iterable
        column = [[item] if not isinstance(item, collections.abc.Iterable) or isinstance(item, (str, bytes)) else item for item in column]
        return list(itertools.chain.from_iterable(column))

    # def get_initial_size(self):
    #     def fill_most_frequent(group):
    #         for column in ['initial_size_a', 'initial_size_b', 'initial_size_c']:
    #             column_values = group[column].dropna()
                
    #             if not column_values.empty:
    #                 # Check if the first non-null value is a list
    #                 if isinstance(column_values.iloc[0], list):
    #                     flattened_values = self.flatten_list(column_values)
    #                     most_frequent = pd.Series(flattened_values).mode().dropna()
    #                 else:
    #                     # Ensure column_values does not contain arrays or lists
    #                     if column_values.apply(lambda x: isinstance(x, (list, np.ndarray))).any():
    #                         flattened_values = self.flatten_list(column_values)
    #                         most_frequent = pd.Series(flattened_values).mode().dropna()
    #                     else:
    #                         most_frequent = column_values.mode().dropna()
                    
    #                 if not most_frequent.empty:
    #                     group[column] = group[column].fillna(most_frequent.iloc[0]).infer_objects(copy=False)
    #         return group
        
    #     self.df = self.df.groupby('patient_id').apply(fill_most_frequent).reset_index(drop=True)
        
    #     return self

    def get_initial_location(self):
        def fill_most_frequent(group):
            for column in ['location_initial_lesion']:
                column_values = group[column].dropna()
                
                if not column_values.empty:
                    # Check if the first non-null value is a list
                    if isinstance(column_values.iloc[0], list):
                        flattened_values = self.flatten_list(column_values)
                        most_frequent = pd.Series(flattened_values).mode().dropna()
                    else:
                        # Ensure column_values does not contain arrays or lists
                        if column_values.apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                            flattened_values = self.flatten_list(column_values)
                            most_frequent = pd.Series(flattened_values).mode().dropna()
                        else:
                            most_frequent = column_values.mode().dropna()
                    
                    if not most_frequent.empty:
                        group[column] = group[column].fillna(most_frequent.iloc[0]).infer_objects(copy=False)
            return group
        
        self.df = self.df.groupby('patient_id').apply(fill_most_frequent).reset_index(drop=True)
        
        return self

    def get_initial_metastasis(self):
        def fill_most_frequent(group):
            
            most_frequent = group['metastasis_initial_presentation_exists'].mode().dropna()
            if not most_frequent.empty:
                group['metastasis_initial_presentation_exists'] = group['metastasis_initial_presentation_exists'].fillna(most_frequent.iloc[0])
            return group
        
        self.df = self.df.groupby('patient_id').apply(fill_most_frequent).reset_index(drop=True)
        return self

    def get_time_relative(self):
        """
        Computes the difference in days with the previous date_of_sarcomaboard_presentation for each patient.
        The first difference is computed with entry_date.

        Returns:
            self: Updates the DataFrame with the time differences.
        """
        # Sort the DataFrame by patient_id and date_of_sarcomaboard_presentation
        self.df = self.df.sort_values(by=['patient_id', 'date_of_surgery_dd_mm_yyyy'])

        # Compute the difference in days with the previous date_of_sarcomaboard_presentation
        self.df['date_of_surgery_dd_mm_yyyy'] = pd.to_datetime(self.df['date_of_surgery_dd_mm_yyyy'], errors='coerce')
        self.df['entry_date'] = pd.to_datetime(self.df['entry_date'])
        self.df['time_relative_sarcomaboard_presentation'] = self.df.groupby('patient_id').apply(
                lambda group: (group['date_of_surgery_dd_mm_yyyy'] - group['entry_date']).dt.days
            ).reset_index(drop=True)

        # Ensure the column is of integer type
        self.df['time_relative_sarcomaboard_presentation'] = self.df['time_relative_sarcomaboard_presentation'].astype('Int64')

        return self

    def fill_most_frequent(self, group, column):
        if any(isinstance(item, list) for item in group[column]):
            return group
        most_frequent = group[column].dropna().mode()
        if not most_frequent.empty:
            group[column] = most_frequent.iloc[0]
        return group

    def get_general_features(self):
        for column in self.relevant_features['static']['general'].keys():
            try:
                if column not in self.df.columns:
                    continue
                self.df = self.df.groupby('patient_id').apply(lambda group: self.fill_most_frequent(group, column)).reset_index(drop=True)
            except Exception as e:
                print(f"Error processing column {column}: {e}")
                continue
        return self
    
    def get_tumor_characteristics(self):
        for column in self.relevant_features['static']['tumor_characteristics'].keys():
            if column not in self.df.columns:
                continue
            self.df = self.df.groupby('patient_id').apply(lambda group: self.fill_most_frequent(group, column)).reset_index(drop=True)
        return self

    def get_static_treatments(self):
        for column in self.relevant_features['static']['treatments'].keys():
            if column not in self.df.columns:
                continue
            self.df = self.df.groupby('patient_id').apply(lambda group: self.fill_most_frequent(group, column)).reset_index(drop=True)
        return self



    def post_process(self):
        """
        Post Process the data and return the transformed DataFrame.

        Returns:
            pandas.DataFrame: The transformed DataFrame.
        """
        return (self
                .assign_train_test_status()
                .get_time_relative()  
                .get_first_therapy()
                .get_first_diagnosis()
                .get_first_anatomic_region_code_grouping()
                .get_discontinuation_chemotherapy()
                .get_duration_radiotherapy()
                .get_initial_size()
                .get_initial_location()
                .get_initial_metastasis()
                .df
                )
    
    def mock_post_process(self):
        return (self.df)



