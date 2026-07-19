import pandas as pd
import numpy as np
import re 
import copy
from collections import defaultdict 
import warnings
import json
import os 
from etl_sdt.extract.pdf_extractor import extract_text_from_pdf
import unicodedata
# from langdetect import detect
from datetime import datetime, timezone
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
        
        def extract_code_and_text(text):
            """Extract numeric/code prefix and diagnosis text from WHO codes."""
            text_str = str(text).strip()
            # Match patterns like "5.3.1.", "M.1.", "BL.0", "0 ", etc.
            import re
            code_match = re.match(r'^([0-9]+\.?[0-9]*\.?[0-9]*\.?|[A-Z]+\.[0-9]+\.?|[0-9]+)\s+', text_str)
            if code_match:
                code = code_match.group(1).strip('.')
                text = text_str[code_match.end():].strip()
                return code, text
            return None, text_str
        
        def extract_base_code(text):
            """Extract just the first letter and number (e.g., 'D18.4.' -> 'D18', 'B10.1' -> 'B10')."""
            text_str = str(text).strip()
            # Match letter followed by digits, ignoring dots and anything after
            import re
            base_match = re.match(r'^([A-Z]\d+)', text_str)
            if base_match:
                return base_match.group(1)
            return None
        
        def has_code_match(val, partial):
            """Match based on codes - first try base code (letter+number), then fall back to text."""
            # Try matching base codes first (e.g., D18, B10, S12)
            val_base = extract_base_code(val)
            partial_base = extract_base_code(partial)
            
            if val_base and partial_base and val_base == partial_base:
                return True
            
            # If base codes don't match or don't exist, try text matching
            val_code, val_text = extract_code_and_text(val)
            partial_code, partial_text = extract_code_and_text(partial)
            
            if val_text and partial_text:
                val_text_lower = val_text.lower()
                partial_text_lower = partial_text.lower()
                
                # Check substring match
                if partial_text_lower in val_text_lower:
                    return True
                
                # Check token overlap for multi-word descriptions
                val_tokens = set(val_text_lower.split())
                partial_tokens = set(partial_text_lower.split())
                # Require significant overlap (more than 50% of partial tokens)
                if len(partial_tokens) > 0:
                    overlap = len(val_tokens & partial_tokens)
                    if overlap / len(partial_tokens) > 0.5:
                        return True
            
            return False

        for col, config in label_dict.items():
            # Some label dictionaries may contain optional or stale keys
            # that are not present in the current dataframe.
            if col not in df.columns:
                continue

            mapping = config["mapping"]
            mode = config.get("mode", "substring")

            def match(val):
                val_str = str(val).lower()
                for partial, mapped in mapping.items():
                    partial_str = str(partial).lower()
                    if (
                        (mode == "substring" and partial_str in val_str) or
                        (mode == "token" and has_token_overlap(val_str, partial_str)) or
                        (mode == "code" and has_code_match(str(val), str(partial)))
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

        def get_first_valid_date_value(value):
            """Return a scalar date-like value usable by episode parsing, else None."""
            if isinstance(value, dict):
                if "$date" in value:
                    return value
                return None

            if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                for item in value:
                    candidate = get_first_valid_date_value(item)
                    if candidate is not None:
                        return candidate
                return None

            if not self._is_valid_value(value):
                return None

            if isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
                return value

            if isinstance(value, str):
                try:
                    parsed = pd.to_datetime(value, errors='coerce')
                    if pd.notnull(parsed):
                        return value
                except Exception:
                    return None

            return None

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
        
        def parse_date_value(value):
            """Parse date values from datetime, ISO strings, pandas timestamps, and MongoDB Extended JSON."""
            if value is None:
                return None

            if isinstance(value, dict):
                # MongoDB extended JSON dates are usually encoded as {"$date": ...}
                if "$date" in value:
                    return parse_date_value(value.get("$date"))
                return None

            if isinstance(value, pd.Timestamp):
                dt = value.to_pydatetime()
            elif isinstance(value, np.datetime64):
                dt = pd.to_datetime(value).to_pydatetime()
            elif isinstance(value, datetime):
                dt = value
            elif isinstance(value, str):
                value = value.strip()
                if not value:
                    return None
                try:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except Exception:
                    try:
                        parsed = pd.to_datetime(value, utc=True)
                        if pd.isna(parsed):
                            return None
                        dt = parsed.to_pydatetime()
                    except Exception:
                        return None
            else:
                return None

            # Normalize to naive UTC for consistent datetime comparisons.
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt

        def get_treatment_date(treatment):
            # Collect all valid dates from the treatment
            for key in ['date_field', 'date_sarcomaboard']:
                if key in treatment and treatment[key]:
                    parsed_date = parse_date_value(treatment[key])
                    if parsed_date is not None:
                        return parsed_date
            return None  
        
        def parse_follow_up_date(follow_up_date):
            """Parse follow-up date from various formats"""
            return parse_date_value(follow_up_date)
        
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
            
            # Get the baseline date from pathology report when present.
            baseline_date = None
            if patient_static_data and patient_static_data.get('general', {}).get('date_pathology_report'):
                pathology_date = patient_static_data['general']['date_pathology_report']
                baseline_date = parse_follow_up_date(pathology_date)
            
            # Ensure episodes start early enough to include interventions occurring before pathology date.
            earliest_treatment_date = treatments_with_date[0]['_parsed_date']
            if baseline_date:
                baseline_date = min(baseline_date, earliest_treatment_date)
            else:
                baseline_date = earliest_treatment_date
            
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
                
                # Calculate actual duration in days
                duration_days = (current_end - current_start).days + 1  # +1 to include both start and end day
                
                # Find treatments in this 6-month period
                episode_treatments = []
                for treatment in treatments_with_date:
                    if current_start <= treatment['_parsed_date'] <= current_end:
                        # Deep copy to avoid reference issues
                        episode_treatments.append(treatment.copy())
                
                # Create episode
                episode = {
                    "episode_id": episode_id,
                    "start_date": current_start,
                    "end_date": current_end,
                    "duration_days": duration_days,
                    "n_interventions": len(episode_treatments),
                    "treatments": episode_treatments
                }
                
                # If episode has treatments but no diagnosis, copy diagnosis from previous episode
                if episode_treatments and episodes:
                    has_diagnosis = any(t.get('section') == 'diagnosis' for t in episode_treatments)
                    if not has_diagnosis:
                        # Find the most recent diagnosis
                        for prev_ep in reversed(episodes):
                            if prev_ep["treatments"]:
                                for treatment in prev_ep["treatments"]:
                                    if treatment.get('section') == 'diagnosis':
                                        # Copy diagnosis with updated date
                                        copied_diagnosis = copy.deepcopy(treatment)
                                        copied_diagnosis['_is_copied'] = True
                                        if 'date_field' in copied_diagnosis:
                                            copied_diagnosis['date_field'] = current_start.isoformat()
                                        if 'date_sarcomaboard' in copied_diagnosis:
                                            copied_diagnosis['date_sarcomaboard'] = current_start.isoformat()
                                        copied_diagnosis['_parsed_date'] = current_start
                                        episode["treatments"].insert(0, copied_diagnosis)
                                        episode["n_interventions"] = len(episode["treatments"])
                                        break
                                break
                
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
                                t.get('section') not in ['diagnosis'] or 
                                not t.get('_is_copied', False) for t in prev_ep["treatments"]
                            ):
                                previous_episode = prev_ep
                                break
                        
                        if previous_episode:
                            episode["treatments"] = []
                            for treatment in previous_episode["treatments"]:
                                # Only copy diagnosis sections
                                if treatment.get('section') in ['diagnosis']:
                                    # Deep copy to preserve nested fields
                                    copied_treatment = copy.deepcopy(treatment)
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
        # Also apply data_assembler to each episode
        for i, episode in enumerate(episodes):
            episode['start_date'] = episode['start_date'].isoformat().replace("+00:00", "Z")
            episode['end_date'] = episode['end_date'].isoformat().replace("+00:00", "Z")
            for treatment in episode['treatments']:
                treatment.pop('_parsed_date', None)
            
            # Apply data_assembler and store result back in the list
            episodes[i] = self.data_assembler(episode)
        
        return episodes

    def group_episodes_into_treatments(self, episodes: list) -> dict:
        def is_non_placeholder(value):
            if value is None:
                return False
            if isinstance(value, str):
                return value.strip().lower() not in {"", "-", "--", "na", "n/a", "none", "nan"}
            return bool(pd.notnull(value))

        def parse_any_date(value):
            """Parse scalar/list/Mongo-date values to naive datetime when possible."""
            if value is None:
                return None

            if isinstance(value, dict):
                if "$date" in value:
                    return parse_any_date(value.get("$date"))
                return None

            if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                for item in value:
                    parsed = parse_any_date(item)
                    if parsed is not None:
                        return parsed
                return None

            if isinstance(value, datetime):
                return value.replace(tzinfo=None) if value.tzinfo is not None else value

            if isinstance(value, pd.Timestamp):
                dt = value.to_pydatetime()
                return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt

            if isinstance(value, str):
                try:
                    dt = pd.to_datetime(value, errors="coerce", utc=True)
                    if pd.notnull(dt):
                        return dt.to_pydatetime().replace(tzinfo=None)
                except Exception:
                    return None

            return None

        treatments = {
            "any_surgery":               0,
            "any_chemotherapy":          0,
            "any_radiotherapy":          0,
            "radiotherapy_preoperative": 0,
            "any_metastasis":            0,
            "any_local_recurrence":      0,
            "local_recurrence_initial_diagnosis_timerange": None,
            "any_whoops":                0,
            "dod":                       0,
        }

        # Prefer definitive/index surgery dates (those carrying surgery_indication).
        # Fallback to any surgery date only when definitive dates are unavailable.
        earliest_definitive_surgery_date = None
        earliest_any_surgery_date = None
        for episode in episodes:
            for treatment in episode.get("treatments", []):
                if treatment.get("section") != "surgery":
                    continue
                s_date = parse_any_date(treatment.get("date_field"))
                if s_date is None:
                    continue

                if earliest_any_surgery_date is None or s_date < earliest_any_surgery_date:
                    earliest_any_surgery_date = s_date

                if is_non_placeholder(treatment.get("surgery_indication")):
                    if earliest_definitive_surgery_date is None or s_date < earliest_definitive_surgery_date:
                        earliest_definitive_surgery_date = s_date

        reference_surgery_date = (
            earliest_definitive_surgery_date
            if earliest_definitive_surgery_date is not None
            else earliest_any_surgery_date
        )

        earliest_local_recurrence_days = None

        for episode in episodes:
            if episode.get("surgery", 0):
                treatments["any_surgery"] = 1
            if episode.get("chemotherapy", 0):
                treatments["any_chemotherapy"] = 1
            if episode.get("metastasis", 0):
                treatments["any_metastasis"] = 1
            if episode.get("local_recurrence", 0):
                treatments["any_local_recurrence"] = 1
            if episode.get("radiotherapy", 0):
                treatments["any_radiotherapy"] = 1

                for treatment in episode.get("treatments", []):
                    if treatment.get("section") != "radiotherapy":
                        continue

                    rt_indication = treatment.get("radiotherapy_indication")
                    if isinstance(rt_indication, (list, tuple, set, np.ndarray, pd.Series)):
                        rt_indication_values = {
                            str(val).strip().lower()
                            for val in rt_indication
                            if pd.notnull(val)
                        }
                        is_preoperative = "preoperative" in rt_indication_values
                    else:
                        is_preoperative = str(rt_indication).strip().lower() == "preoperative"

                    # Fallback for unmapped/empty indication (e.g., "nicht mappen"):
                    # infer preoperative if RT ends (or starts) on/before surgery date.
                    if not is_preoperative and reference_surgery_date is not None:
                        rt_end = parse_any_date(treatment.get("radiotherapy_end_date"))
                        rt_start = parse_any_date(treatment.get("radiotherapy_start_date"))
                        rt_anchor = rt_end if rt_end is not None else rt_start
                        if rt_anchor is not None and rt_anchor <= reference_surgery_date:
                            is_preoperative = True

                    if is_preoperative:
                        treatments["radiotherapy_preoperative"] = 1
                        break

            # WHOOPS detection: if any treatment in the episode contains WHOOPS evidence, set flag
            # Check for explicit `whoops` flag or presence of WHOOPS-specific fields
            for tr in episode.get("treatments", []):
                if tr.get("whoops", 0) == 1:
                    treatments["any_whoops"] = 1
                    break
                if any(pd.notnull(tr.get(f)) for f in ("date_whoops", "whoops_margin_status", "whoops_surgery_institution")):
                    treatments["any_whoops"] = 1
                    break

            # DOD is stored in the diagnosis list under section "follow_up"
            for diag in episode.get("diagnosis", []):
                if diag.get("section") == "local_recurrence":
                    lr_days = diag.get("local_recurrence_initial_diagnosis_timerange")
                    if is_non_placeholder(lr_days):
                        try:
                            lr_days = int(lr_days)
                        except (TypeError, ValueError):
                            lr_days = None
                        if lr_days is not None and (
                            earliest_local_recurrence_days is None
                            or lr_days < earliest_local_recurrence_days
                        ):
                            earliest_local_recurrence_days = lr_days

                if (diag.get("section") == "follow_up" and
                        diag.get("last_status") == "DOD"):
                    treatments["dod"] = 1

        if treatments["any_local_recurrence"]:
            treatments["local_recurrence_initial_diagnosis_timerange"] = earliest_local_recurrence_days

        return treatments

    def add_preoperative_rt_margin(self, patient_record: dict) -> dict:
        treatments = patient_record.setdefault("treatments", {})
        margin = treatments.get("pathologist_margin_judgement")
        if not self._is_valid_value(margin):
            margin = treatments.get("pathologist_margin_judgment")

        if not self._is_valid_value(margin):
            return patient_record

        treatments.setdefault("pathologist_margin_judgment", margin)
        if treatments.get("radiotherapy_preoperative") == 1:
            treatments["pathologist_margin_judgment_pre_rt"] = margin

        return patient_record


    def data_assembler(self, episode):
        # Group treatments by section and merge dictionaries for the same section
        treatments = episode.get("treatments", [])
        episode["surgery"]          = 0
        episode["chemotherapy"]     = 0
        episode["radiotherapy"]     = 0
        episode["metastasis"]       = 0
        episode["local_recurrence"] = 0
        episode["final_status"]     = 0
        
        # Always initialize diagnosis as an empty array
        episode["diagnosis"] = []
        
        if treatments:
            merged = {}
            for treatment in treatments:
                section = treatment.get("section")
                # assign the treatments presence
                # Count systemic_therapy as chemotherapy
                if section == "systemic_therapy":
                    episode["chemotherapy"] = 1
                elif section == "local_recurrence":
                    episode["local_recurrence"] = 1
                    local_recurrence_treatment = str(
                        treatment.get("treatment_local_recurrence", "")
                    ).lower()
                    # Local recurrence can encode treatment modality; map chemo mentions to episode flag.
                    if "chemo" in local_recurrence_treatment:
                        episode["chemotherapy"] = 1
                elif section == "follow_up":
                    episode["final_status"] = 1
                elif section in ["surgery", "chemotherapy", "radiotherapy", "metastasis"]:
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
                                        # Recursively merge the nested dict
                                        for nested_key, nested_value in value.items():
                                            if nested_key in merged_val:
                                                # If both nested values are lists, deduplicate when merging
                                                if isinstance(merged_val[nested_key], list) and isinstance(nested_value, list):
                                                    # Combine and remove duplicates while preserving order
                                                    combined = merged_val[nested_key] + nested_value
                                                    merged_val[nested_key] = list(dict.fromkeys(combined))
                                                else:
                                                    merged_val[nested_key] = nested_value
                                            else:
                                                merged_val[nested_key] = nested_value
                                        merged[section][key] = merged_val
                                    # If both values are lists, concatenate and remove duplicates
                                    elif isinstance(merged[section][key], list) and isinstance(value, list):
                                        # Combine lists and remove duplicates while preserving order
                                        combined = merged[section][key] + value
                                        merged[section][key] = list(dict.fromkeys(combined))
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
            # Include all treatment-related sections
            if sec in ['surgery', 'chemotherapy', 'radiotherapy', 'systemic_therapy']:
                treatments.append(item)
            elif sec in ['diagnosis', 'metastasis', 'follow_up', 'local_recurrence']:
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
    
    def _is_valid_value(self, val):
        """
        Checks if a value is valid (non-null), handling both lists and single values.
        
        Args:
            val: The value to check (can be a list or single value)
        
        Returns:
            bool: True if the value is valid (non-null), False otherwise
        """
        if isinstance(val, list):
            # For lists, check if not empty and has at least one non-null value
            return len(val) > 0 and any(pd.notnull(v) for v in val)
        elif isinstance(val, pd.Series):
            # For Series, check if any values are non-null
            return not val.empty and pd.notnull(val).any()
        else:
            # For single values, use pandas notnull check
            return pd.notnull(val)

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

        def get_first_valid_date_value(value):
            """Return a scalar date-like value usable by episode parsing, else None."""
            if isinstance(value, dict):
                if "$date" in value:
                    return value
                return None

            if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                for item in value:
                    candidate = get_first_valid_date_value(item)
                    if candidate is not None:
                        return candidate
                return None

            if not self._is_valid_value(value):
                return None

            if isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
                return value

            if isinstance(value, str):
                try:
                    parsed = pd.to_datetime(value, errors='coerce')
                    if pd.notnull(parsed):
                        return value
                except Exception:
                    return None

            return None

        records = []
        
        for group_idx, (patient_id, group_df) in enumerate(df.groupby(key_column), start=1):
            
            patient_record = {"_id": patient_id}
            
            row = group_df.iloc[-1]

            # Extract static features - each group_name has a list of field names
            for group_name, field_list in relevant_features['static'].items():
                patient_record[group_name] = {}
                for field_name in field_list:
                    val = row.get(field_name)
                    if self._is_valid_value(val):
                        patient_record[group_name][field_name] = val

            #TODO: adapt 
            # group_df = self._sort_by_column(group_df, column=['patient_id', 'time_relative_sarcomaboard_presentation'], ascending=True)
                        
            patient_record['diagnosis_treatment_sequence'] = []
            
            # Collect diagnosis once from the first row (diagnosis is patient-level, not treatment-specific)
            diagnosis_added = False
            
            for idx, (index, row) in enumerate(group_df.iterrows(), start=1):
                             
                for group, config in relevant_features['dynamic'].items():
                    # Extract the fields from the config
                    field_list = config.get('fields', [])
                    
                    # Build dictionary with fields that have non-null values
                    dict_tmp = {}
                    for field_name in field_list:
                        val = row.get(field_name)
                        if self._is_valid_value(val):
                            dict_tmp[field_name] = val
                    
                    # Also extract date_field and date_sarcoma_board if present
                    if 'date_field' in config:
                        date_val = row.get(config['date_field'])
                        parsed_date_val = get_first_valid_date_value(date_val)
                        if parsed_date_val is not None:
                            dict_tmp['date_field'] = parsed_date_val
                        elif group == 'surgery':
                            # Whoops-only cases may not have date_index_surgery; use whoops date as surgery date.
                            fallback_surgery_date = row.get('date_whoops')
                            fallback_surgery_parsed_date = get_first_valid_date_value(fallback_surgery_date)
                            if fallback_surgery_parsed_date is not None:
                                dict_tmp['date_field'] = fallback_surgery_parsed_date
                        elif group == 'radiotherapy':
                            # Some frozen exports carry RT evidence without explicit start date.
                            # Anchor such RT records to pathology date so they can be assigned to an episode.
                            fallback_rt_date = row.get('date_pathology_report')
                            fallback_parsed_date = get_first_valid_date_value(fallback_rt_date)
                            if fallback_parsed_date is not None:
                                dict_tmp['date_field'] = fallback_parsed_date
                        elif group == 'systemic_therapy':
                            # Some systemic rows only contain the cycle end date.
                            fallback_systemic_date = row.get('cycle_end_date')
                            fallback_parsed_date = get_first_valid_date_value(fallback_systemic_date)
                            if fallback_parsed_date is not None:
                                dict_tmp['date_field'] = fallback_parsed_date
                    
                    if 'date_sarcoma_board' in config:
                        date_sb = row.get(config['date_sarcoma_board'])
                        if self._is_valid_value(date_sb):
                            dict_tmp['date_sarcoma_board'] = date_sb
                    
                    # Only add if there are fields with values
                    if dict_tmp:
                        # Check flags for certain sections
                        should_include = False
                        
                        # Diagnosis should only be added once from the first row
                        if group == 'diagnosis' and not diagnosis_added:
                            should_include = True
                            diagnosis_added = True
                        elif group == 'surgery' and row.get('surgery_flag', None) == 1:
                            should_include = True
                        elif group == 'systemic_therapy' and row.get('chemotherapy_flag', None) == 1:
                            should_include = True
                        elif group in ['radiotherapy'] and row.get('radiation_oncology_flag', None) == 1:
                            should_include = True
                        elif group == 'local_recurrence' and row.get('local_recurrence_flag', None) == 1:
                            should_include = True
                        elif group == 'follow_up' and row.get('final_status_flag', None) == 1:
                            should_include = True
                        elif group in ['radiology', 'events', 'recurrence_metastasis']:
                            should_include = True
                        elif group in ['metastasis']:
                            # Metastasis - always include if it has data
                            should_include = True
                        elif row.get('metastasis_flag', None) == 1:
                            should_include = True
                        
                        if should_include:
                            dict_tmp['section'] = group
                            patient_record['diagnosis_treatment_sequence'].append(dict_tmp)
            
            # Extract follow-up date for timeline episodes
            follow_up_date = None
            if timeline and patient_record.get('general', {}).get('date_last_follow_up'):
                follow_up_date = patient_record['general']['date_last_follow_up']
                    
            patient_record['episodes'] = self.episode_restructure_data(
                patient_record['diagnosis_treatment_sequence'], 
                timeline=timeline, 
                follow_up_date=follow_up_date,
                patient_static_data=patient_record  # Pass the entire patient record for static data access
            )
            patient_record['treatments'].update(self.group_episodes_into_treatments(patient_record['episodes']))
            self.add_preoperative_rt_margin(patient_record)
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
            # Handle pandas Series objects (shouldn't normally occur but can happen with nested data)
            if isinstance(value, pd.Series):
                # If it's a Series with one element, extract the scalar value
                if len(value) == 1:
                    return split_nested_values(value.iloc[0])
                # If it's a Series with multiple elements, convert to list and process
                return [split_nested_values(item) for item in value.tolist()]
            
            # Check for list first to avoid pd.isna() ambiguity with lists
            if isinstance(value, list):
                # If already a list, recursively process each element
                return [split_nested_values(item) for item in value]
            
            if pd.isna(value):
                return value
            
            if isinstance(value, str):
                # Check if value contains any of the separators
                if ';' in value or '|' in value:
                    # Split by any of the separators and clean up whitespace
                    # Use regex to split by multiple separators
                    import re
                    parts = re.split(r'[;|]', value)
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
                # df[column] = df[column].astype('object')
        
        return df

class FeatureExtractor:
    """
    This class is responsible for extracting relevant features from a DataFrame.
    """

    UPPER_EXTREMITY_LOCATION_CODES = frozenset({
        "clavicle",
        "scapula",
        "shoulder_girdle",
        "scapular_region",
        "axilla",
        "shoulder_joint",
        "humerus",
        "upper_arm",
        "elbow",
        "elbow_region",
        "forearm_bones",
        "forearm",
        "wrist",
        "hand",
    })
    LOWER_EXTREMITY_LOCATION_CODES = frozenset({
        "hip_joint",
        "thigh",
        "femur",
        "knee",
        "patella",
        "lower_leg",
        "tibia_fibula",
        "ankle",
        "foot",
        "inguinal_region",
    })

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
        # Handle Series objects (shouldn't normally occur but can in nested data)
        if isinstance(element, pd.Series):
            if len(element) == 1:
                element = element.iloc[0]
            else:
                # For multi-value Series, convert to list and process each element
                return [self.convert_single_element(data_type, item) for item in element.tolist()]
        
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
        
        self.df['age']  = self.df['date_pathology_report'].dt.year - self.df["year_of_birth"] 

        self.relevant_features['static']['general'].append('age')
        
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
        def has_surgery_evidence(row):
            if pd.notnull(row.get('surgery_indication')):
                return 1

            # Some exports have missing/misaligned surgery_indication but still
            # provide a concrete index surgery date.
            if pd.notnull(row.get('date_index_surgery')):
                return 1

            # Treat whoops surgery as surgery event as well.
            whoops_fields = [
                'date_whoops',
                'whoops_margin_status',
                'whoops_surgery_institution',
            ]
            return 1 if any(pd.notnull(row.get(field)) for field in whoops_fields) else 0

        self.df['surgery_flag'] = self.df.apply(
            has_surgery_evidence,
            axis=1
        )

        self.relevant_features['dynamic']['surgery']['surgery_flag']='surgery_flag'
        self.data_type_mapping['surgery_flag'] = int
        
        #radiation_oncology_flag
        def has_radiotherapy_evidence(row):
            def is_present(value):
                if isinstance(value, dict):
                    # Support Mongo-style extended JSON dates and nested structures.
                    if "$date" in value:
                        return is_present(value.get("$date"))
                    return any(is_present(v) for v in value.values())

                if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                    return any(is_present(v) for v in value)

                if value is None:
                    return False

                if isinstance(value, str):
                    return value.strip().lower() not in {
                        "", "nan", "none", "-", "--", "na", "n/a", "nicht mappen"
                    }

                return bool(pd.notnull(value))

            indication = row.get('radiotherapy_indication')

            # Direct indication evidence, excluding explicit "no radiotherapy".
            if is_present(indication):
                if isinstance(indication, (list, tuple, set, np.ndarray, pd.Series)):
                    indication_values = [
                        str(v).strip().lower()
                        for v in indication
                        if is_present(v)
                    ]
                    if any(v != 'no radiotherapy' for v in indication_values):
                        return 1
                elif str(indication).strip().lower() != 'no radiotherapy':
                    return 1

            # Fallback evidence when indication is unmapped/missing but RT values exist in source sheet.
            evidence_fields = [
                'radiotherapy_start_date',
                'radiotherapy_end_date',
                'radiotherapy_total_dose',
                'radiotherapy_num_fractions',
                'radiotherapy_type',
            ]
            return 1 if any(is_present(row.get(field)) for field in evidence_fields) else 0

        self.df['radiation_oncology_flag'] = self.df.apply(has_radiotherapy_evidence, axis=1)

        self.relevant_features['dynamic']['radiotherapy']['radiation_oncology_flag'] = 'radiation_oncology_flag'
        self.data_type_mapping['radiation_oncology_flag'] = int
        
        #chemotherapy_flag
        def has_systemic_therapy_evidence(row):
            def is_systemic_present(value):
                if isinstance(value, dict):
                    if "$date" in value:
                        return is_systemic_present(value.get("$date"))
                    return any(is_systemic_present(v) for v in value.values())

                if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                    return any(is_systemic_present(v) for v in value)

                if value is None:
                    return False

                if isinstance(value, str):
                    return value.strip().lower() not in {
                        "", "nan", "none", "-", "--", "na", "n/a", "nicht mappen"
                    }

                return bool(pd.notnull(value))

            evidence_fields = [
                'systemic_treatment_reason',
                'line_of_treatment',
                'systemic_therapy_type',
                'drug_name',
                'soft_tissue_protocol_name',
                'bone_protocol_name',
                'clinical_trial',
                'cycle_start_date',
                'cycle_end_date',
                'systemic_therapy_discontinuation_reason',
                'num_cycles_executed',
                'dose_unit',
                'dose_reduction',
                'applied_dose_mg_m2',
                'toxicity_days_of_cycle',
                'toxicity_type',
                'ctcae_grade',
            ]
            return 1 if any(is_systemic_present(row.get(field)) for field in evidence_fields) else 0

        self.df['chemotherapy_flag'] = self.df.apply(has_systemic_therapy_evidence, axis=1)

        self.relevant_features['dynamic']["systemic_therapy"]['chemotherapy_flag'] = 'chemotherapy_flag'
        self.data_type_mapping['chemotherapy_flag'] = int

        #local_recurrence_flag
        self.df['local_recurrence_flag'] = self.df.apply(
            lambda row: 1 if pd.notnull(row['date_local_recurrence']) else 0,
            axis=1
        )

        self.relevant_features['dynamic']['local_recurrence']['local_recurrence_flag'] = 'local_recurrence_flag'
        self.data_type_mapping['local_recurrence_flag'] = int

        #final_status_flag
        self.df['final_status_flag'] = self.df.apply(
            lambda row: 1 if pd.notnull(row['last_status']) else 0,
            axis=1
        )

        self.relevant_features['dynamic']['follow_up']['final_status_flag'] = 'final_status_flag'
        self.data_type_mapping['final_status_flag'] = int


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
        # Defragment DataFrame to avoid performance warning
        self.df = self.df.copy()
        sizes = self.df[['size_a_mm', 'size_b_mm', 'size_c_mm']]

        self.df['initial_size'] = np.sqrt(sizes.pow(2).sum(axis=1, min_count=1))

        self.relevant_features['static']['tumor_characteristics'].append('initial_size')
        self.data_type_mapping['initial_size'] = int
        return self


    def get_initial_pathology_report_date(self):
        """Use the earliest pathology report date as the patient-level initial diagnosis date."""
        if 'patient_id' not in self.df.columns or 'date_pathology_report' not in self.df.columns:
            return self

        self.df['date_pathology_report'] = pd.to_datetime(self.df['date_pathology_report'], errors='coerce')
        initial_dates = self.df.groupby('patient_id')['date_pathology_report'].transform('min')
        self.df['date_pathology_report'] = initial_dates.fillna(self.df['date_pathology_report'])
        return self

    def get_merge_grading_biopsy_resection(self):
        
        # Merge grading information: two-level priority
        # Level 1: Look for G1, G2, or G3 in either column
        # Level 2: If neither has those grades, return any valid string
        def get_valid_grading(biopsy_val, resection_val):
            valid_grades = ['G1', 'G2', 'G3']
            
            # Level 1: Check for G1, G2, or G3 in biopsy first
            if pd.notna(biopsy_val) and biopsy_val in valid_grades:
                return biopsy_val
            # Check for G1, G2, or G3 in resection
            if pd.notna(resection_val) and resection_val in valid_grades:
                return resection_val
            
            # Level 2: If neither has G1/G2/G3, return any valid string from biopsy
            if pd.notna(biopsy_val):
                return biopsy_val
            # Fallback to any valid string from resection
            if pd.notna(resection_val):
                return resection_val
            
            return None
        
        self.df['biopsy_grading'] = self.df.apply(
            lambda row: get_valid_grading(row.get('biopsy_grading'), row.get('resection_grading')),
            axis=1
        )
        return self

    def time_range_in_episodes(self, start_date_col, end_date_col):
        # Ensure both date columns are datetime type
        self.df[start_date_col] = pd.to_datetime(self.df[start_date_col], errors='coerce')
        self.df[end_date_col] = pd.to_datetime(self.df[end_date_col], errors='coerce')
        
        months_diff = (self.df[start_date_col].dt.year - self.df[end_date_col].dt.year) * 12 + \
                  (self.df[start_date_col].dt.month - self.df[end_date_col].dt.month)
        result = months_diff / 6
        return result.fillna(0).replace([np.inf, -np.inf], 0).astype(int)

 
    def time_range_in_days(self, start_date_col, end_date_col, max_days=None):
        # Ensure both date columns are datetime type
        self.df[start_date_col] = pd.to_datetime(self.df[start_date_col], errors='coerce')
        self.df[end_date_col] = pd.to_datetime(self.df[end_date_col], errors='coerce')
        
        days_diff = (self.df[start_date_col] - self.df[end_date_col]).dt.days
        
        # Apply threshold if specified
        if max_days is not None:
            # Log warnings for values exceeding threshold
            excessive_mask = days_diff > max_days
            if excessive_mask.any():
                excessive_count = excessive_mask.sum()
                max_val = days_diff[excessive_mask].max()
                logger.warning(f"{start_date_col}: Found {excessive_count} rows with duration >{max_days} days (max={max_val}). Setting to None.")
                days_diff = days_diff.where(~excessive_mask, None)
            
            # Also check for negative durations
            negative_mask = days_diff < 0
            if negative_mask.any():
                negative_count = negative_mask.sum()
                logger.warning(f"{start_date_col}: Found {negative_count} rows with negative duration. Setting to None.")
                days_diff = days_diff.where(~negative_mask, None)
        
        return days_diff
 

    def get_episodes_whoops(self):
        
        self.df["episodes_whoops"]  = self.time_range_in_episodes('date_whoops', 'date_pathology_report')
        self.df["episodes_surgery"]  = self.time_range_in_episodes('date_index_surgery', 'date_pathology_report')
       
        self.df["episodes_surgery"] = self.df.apply(
            lambda row: min(filter(pd.notna, [row.get("episodes_whoops"), row.get("episodes_surgery")]), default=None)
            if pd.notna(row.get("episodes_whoops")) or pd.notna(row.get("episodes_surgery"))
            else None,
            axis=1
        )

        self.relevant_features['dynamic']['surgery']['episodes_field'] = 'episodes_surgery'
        self.data_type_mapping['episodes_surgery'] = int
        return self

    def get_episodes_systemic(self):
        
        self.df['_systemic_episode_date'] = pd.to_datetime(
            self.df.get('cycle_start_date'),
            errors='coerce'
        ).fillna(pd.to_datetime(self.df.get('cycle_end_date'), errors='coerce'))
        self.df["episodes_systemic"]  = self.time_range_in_episodes('_systemic_episode_date', 'date_pathology_report')

        self.relevant_features['dynamic']['systemic_therapy']['episodes_field'] = 'episodes_systemic'
        self.data_type_mapping['episodes_systemic'] = int
        return self
    
    def get_systemic_timerange(self):
        
        self.df["systemic_timerange"]  = self.time_range_in_days('cycle_end_date', 'cycle_start_date', max_days=90)

        self.relevant_features['dynamic']['systemic_therapy']['fields'].append('systemic_timerange')    
        self.data_type_mapping['systemic_timerange'] = int
        return self

    def get_episodes_radiotherapy(self):
        
        self.df["episodes_radiotherapy"]  = self.time_range_in_episodes('radiotherapy_start_date', 'date_pathology_report')

        self.relevant_features['dynamic']['radiotherapy']['episodes_field'] = 'episodes_radiotherapy'
        self.data_type_mapping['episodes_radiotherapy'] = int
        return self

    def get_radiotherapy_timerange(self):
        
        self.df["radiotherapy_timerange"]  = self.time_range_in_days('radiotherapy_end_date', 'radiotherapy_start_date', max_days=90)

        self.relevant_features['dynamic']['radiotherapy']['fields'].append('radiotherapy_timerange')    
        self.data_type_mapping['radiotherapy_timerange'] = int
        return self
    
    def get_merge_first_radio_into_main(self):
        """
        Merges first_radiotherapy_* columns into their corresponding radiotherapy_* columns.
        For each patient, fills null values in the main columns with values from the first_* columns.
        """
        # Define column mappings
        column_mappings = {
            'first_radiotherapy_start_date': 'radiotherapy_start_date',
            'first_radiotherapy_end_date': 'radiotherapy_end_date',
            'first_radiotherapy_indication': 'radiotherapy_indication',
            'first_radiotherapy_type': 'radiotherapy_type',
            'first_radiotherapy_num_fractions': 'radiotherapy_num_fractions',
            'first_radiotherapy_total_dose': 'radiotherapy_total_dose'
        }
        
        # Merge each column pair
        for first_col, main_col in column_mappings.items():
            if first_col in self.df.columns and main_col in self.df.columns:
                # Fill null values in main column with values from first_* column
                self.df[main_col] = self.df[main_col].fillna(self.df[first_col])
            elif first_col in self.df.columns:
                # If main column doesn't exist, create it from first_* column
                self.df[main_col] = self.df[first_col]
        
        return self

    def get_merge_radiotherapy_indication_raw(self):
        """
        Combines raw radiotherapy indication into canonical radiotherapy_indication.
        Keeps the canonical field as the single source used by downstream logic.
        """
        if 'radiotherapy_indication_raw' not in self.df.columns:
            return self

        if 'radiotherapy_indication' not in self.df.columns:
            self.df['radiotherapy_indication'] = None

        placeholder_values = {
            '', '-', '--', 'na', 'n/a', 'none', 'nan', 'nicht mappen'
        }

        def normalize_indication(value):
            if value is None:
                return None

            if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                normalized_values = [normalize_indication(v) for v in value]
                normalized_values = [v for v in normalized_values if v is not None]
                return normalized_values[0] if normalized_values else None

            val_str = str(value).strip().lower()
            if val_str in placeholder_values:
                return None

            if 'preoperative' in val_str or val_str.startswith('[1]') or val_str.startswith('1 '):
                return 'preoperative'
            if 'postoperative' in val_str or val_str.startswith('[2]') or val_str.startswith('2 '):
                return 'postoperative'
            if 'definitive' in val_str or val_str.startswith('[3]') or val_str.startswith('3 '):
                return 'definitive'
            if 'palliative' in val_str or 'palliativ' in val_str or val_str.startswith('[4]') or val_str.startswith('4 '):
                return 'palliative'
            if 'intraoperativ' in val_str or 'intraoperative' in val_str:
                return 'intraoperative'
            if 'no radiotherapy' in val_str or 'no therapy' in val_str:
                return 'no radiotherapy'
            if 'other' in val_str or 'unknown' in val_str:
                return 'other_unknown'

            return str(value).strip()

        def is_missing_canonical(value):
            if value is None:
                return True
            if isinstance(value, float) and pd.isna(value):
                return True
            val_str = str(value).strip().lower()
            return val_str in placeholder_values

        self.df['radiotherapy_indication_raw'] = self.df['radiotherapy_indication_raw'].apply(normalize_indication)

        self.df['radiotherapy_indication'] = self.df.apply(
            lambda row: (
                row.get('radiotherapy_indication_raw')
                if is_missing_canonical(row.get('radiotherapy_indication'))
                else row.get('radiotherapy_indication')
            ),
            axis=1,
        )

        return self
        
    
    def get_expand_local_recurrence(self):
        """
        Expands local recurrence data from numbered suffixes (_1, _2) into multiple rows.
        Creates unified columns: date_local_recurrence, treatment_local_recurrence, date_local_recurrence_surgery
        Preserves ALL patients - patients without recurrence get empty columns.
        """
        recurrence_columns = [
            ('date_local_recurrence_1', 'treatment_local_recurrence_1', 'date_local_recurrence_surgery_1'),
            ('date_local_recurrence_2', 'treatment_local_recurrence_2', 'date_local_recurrence_surgery_2')
        ]
        
        expanded_rows = []
        processed_indices = set()  # Track which patients have recurrence data
        
        for idx, row in self.df.iterrows():
            has_recurrence = False
            for date_col, treatment_col, surgery_col in recurrence_columns:
                if date_col in self.df.columns and pd.notna(row[date_col]):
                    has_recurrence = True
                    new_row = row.copy()
                    new_row['date_local_recurrence'] = row[date_col]
                    new_row['treatment_local_recurrence'] = row.get(treatment_col) if treatment_col in self.df.columns else None
                    new_row['date_local_recurrence_surgery'] = row.get(surgery_col) if surgery_col in self.df.columns else None
                    expanded_rows.append(new_row)
            
            if has_recurrence:
                processed_indices.add(idx)
        
        # Prepare columns to drop
        cols_to_drop = [col for recurrence in recurrence_columns for col in recurrence if col in self.df.columns]
        
        if expanded_rows:
            # Create DataFrame from expanded rows (patients WITH recurrence)
            expanded_df = pd.DataFrame(expanded_rows)
            expanded_df = expanded_df.drop(columns=cols_to_drop, errors='ignore')
            
            # Get patients WITHOUT recurrence data
            patients_without_recurrence = self.df[~self.df.index.isin(processed_indices)].copy()
            if not patients_without_recurrence.empty:
                patients_without_recurrence['date_local_recurrence'] = None
                patients_without_recurrence['treatment_local_recurrence'] = None
                patients_without_recurrence['date_local_recurrence_surgery'] = None
                patients_without_recurrence = patients_without_recurrence.drop(columns=cols_to_drop, errors='ignore')
            
            # Combine both: patients WITH recurrence + patients WITHOUT recurrence
            self.df = pd.concat([expanded_df, patients_without_recurrence], ignore_index=True)
        else:
            # No recurrence data for ANY patient - create empty columns for all
            self.df['date_local_recurrence'] = None
            self.df['treatment_local_recurrence'] = None
            self.df['date_local_recurrence_surgery'] = None
        
        return self
    
    def get_expand_metastasis(self):
        """
        Expands metastasis data from separate pulmonary/extrapulmonary columns into unified structure.
        Creates columns: date_metastasis, metastasis_type, site_extrapulmonary_metastasis
        Preserves ALL patients - patients without metastasis get empty columns.
        """
        expanded_rows = []
        processed_indices = set()  # Track which patients have metastasis data
        
        for idx, row in self.df.iterrows():
            has_metastasis = False
            
            # Handle pulmonary metastasis
            if 'date_pulmonary_metastasis' in self.df.columns and pd.notna(row['date_pulmonary_metastasis']):
                has_metastasis = True
                new_row = row.copy()
                new_row['date_metastasis'] = row['date_pulmonary_metastasis']
                new_row['metastasis_type'] = 'pulmonary'
                new_row['site_extrapulmonary_metastasis'] = None
                expanded_rows.append(new_row)
            
            # Handle extrapulmonary metastasis
            if 'date_extrapulmonary_metastasis' in self.df.columns and pd.notna(row['date_extrapulmonary_metastasis']):
                has_metastasis = True
                new_row = row.copy()
                new_row['date_metastasis'] = row['date_extrapulmonary_metastasis']
                new_row['metastasis_type'] = 'extrapulmonary'
                new_row['site_extrapulmonary_metastasis'] = row.get('site_extrapulmonary_metastasis')
                expanded_rows.append(new_row)
            
            if has_metastasis:
                processed_indices.add(idx)
        
        # Prepare columns to drop
        cols_to_drop = ['date_pulmonary_metastasis', 'date_extrapulmonary_metastasis']
        
        if expanded_rows:
            # Create DataFrame from expanded rows (patients WITH metastasis)
            expanded_df = pd.DataFrame(expanded_rows)
            expanded_df = expanded_df.drop(columns=[col for col in cols_to_drop if col in expanded_df.columns], errors='ignore')
            
            # Get patients WITHOUT metastasis data
            patients_without_metastasis = self.df[~self.df.index.isin(processed_indices)].copy()
            if not patients_without_metastasis.empty:
                patients_without_metastasis['date_metastasis'] = None
                patients_without_metastasis['metastasis_type'] = None
                if 'site_extrapulmonary_metastasis' not in patients_without_metastasis.columns:
                    patients_without_metastasis['site_extrapulmonary_metastasis'] = None
                patients_without_metastasis = patients_without_metastasis.drop(columns=[col for col in cols_to_drop if col in patients_without_metastasis.columns], errors='ignore')
            
            # Combine both: patients WITH metastasis + patients WITHOUT metastasis
            self.df = pd.concat([expanded_df, patients_without_metastasis], ignore_index=True)
        else:
            # No metastasis data for ANY patient - create empty columns for all
            self.df['date_metastasis'] = None
            self.df['metastasis_type'] = None
            if 'site_extrapulmonary_metastasis' not in self.df.columns:
                self.df['site_extrapulmonary_metastasis'] = None
        
        return self
    
    def get_episodes_local_recurrence(self):
        """
        Creates episode numbers for local recurrence events based on date_local_recurrence.
        """
        self.df["episodes_local_recurrence"] = self.time_range_in_episodes('date_local_recurrence', 'date_pathology_report')
        
        self.relevant_features['dynamic']['local_recurrence']['episodes_field'] = 'episodes_local_recurrence'
        self.data_type_mapping['episodes_local_recurrence'] = int
        return self
    
    def get_local_recurrence_initial_diagnosis_timerange(self):
        """
        Calculates days from initial diagnosis/pathology date to local recurrence.
        """
        self.df["local_recurrence_initial_diagnosis_timerange"] = self.time_range_in_days(
            'date_local_recurrence',
            'date_pathology_report'
        )
        self.df["local_recurrence_initial_diagnosis_timerange"] = self.df[
            "local_recurrence_initial_diagnosis_timerange"
        ].where(self.df["local_recurrence_initial_diagnosis_timerange"] >= 0, None)

        lr_fields = self.relevant_features['dynamic']['local_recurrence']['fields']
        if 'local_recurrence_initial_diagnosis_timerange' not in lr_fields:
            lr_fields.append('local_recurrence_initial_diagnosis_timerange')
        self.data_type_mapping['local_recurrence_initial_diagnosis_timerange'] = int
        return self
    
    def get_episodes_metastasis(self):
        """
        Creates episode numbers for metastasis events based on date_metastasis.
        """
        self.df["episodes_metastasis"] = self.time_range_in_episodes('date_metastasis', 'date_pathology_report')
        
        self.relevant_features['dynamic']['metastasis']['episodes_field'] = 'episodes_metastasis'
        self.data_type_mapping['episodes_metastasis'] = int
        return self
    
    def get_episodes_follow_up(self):
        """
        Creates episode numbers for follow-up visits based on date_last_follow_up.
        """
        self.df["episodes_follow_up"] = self.time_range_in_episodes('date_last_follow_up', 'date_pathology_report')
        
        self.relevant_features['dynamic']['follow_up']['episodes_field'] = 'episodes_follow_up'
        self.data_type_mapping['episodes_follow_up'] = int
        return self

    def remove_duplicates_drug_names(self):
        def remove_duplicates(val):
            if isinstance(val, list):
                return list(set(val))
            return val

        self.df['drug_name'] = self.df['drug_name'].apply(remove_duplicates)
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
    
    def _add_static_feature(self, group_name, field_name):
        features = self.relevant_features.get('static', {}).get(group_name)
        if isinstance(features, list) and field_name not in features:
            features.append(field_name)
        elif isinstance(features, dict) and field_name not in features:
            features[field_name] = field_name

    def _add_dynamic_field(self, group_name, field_name):
        group = self.relevant_features.get('dynamic', {}).get(group_name, {})
        fields = group.get('fields')
        if isinstance(fields, list) and field_name not in fields:
            fields.append(field_name)
        elif isinstance(fields, dict) and field_name not in fields:
            fields[field_name] = field_name

    def get_margin_aliases(self):
        if 'pathologist_margin_judgement' in self.df.columns:
            self.df['pathologist_margin_judgment'] = self.df['pathologist_margin_judgement']
        elif 'pathologist_margin_judgment' not in self.df.columns:
            self.df['pathologist_margin_judgment'] = np.nan

        self._add_static_feature('treatments', 'pathologist_margin_judgment')
        self._add_dynamic_field('surgery', 'pathologist_margin_judgment')
        if self.data_type_mapping is not None:
            self.data_type_mapping['pathologist_margin_judgment'] = str
            self.data_type_mapping['pathologist_margin_judgment_pre_rt'] = str
        return self

    @classmethod
    def _map_body_location_group(cls, value):
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            for item in value:
                mapped = cls._map_body_location_group(item)
                if mapped is not None:
                    return mapped
            return None

        if value is None or pd.isna(value):
            return None

        location_code = str(value).strip().lower()
        if not location_code or location_code == 'unknown':
            return None
        if location_code in cls.UPPER_EXTREMITY_LOCATION_CODES:
            return 'upper_extremity'
        if location_code in cls.LOWER_EXTREMITY_LOCATION_CODES:
            return 'lower_extremity'
        return 'trunk'

    def get_body_location_group(self):
        if 'anatomic_region_code' not in self.df.columns:
            return self

        self.df['body_location_group'] = self.df['anatomic_region_code'].apply(self._map_body_location_group)
        self._add_static_feature('tumor_characteristics', 'body_location_group')
        if self.data_type_mapping is not None:
            self.data_type_mapping['body_location_group'] = str
        return self

    def get_extremity_tumor(self):
        if 'anatomic_region_code' not in self.df.columns:
            return self

        self.df['extremity_tumor'] = self.df['anatomic_region_code'].apply(
            lambda value: 1 if self._map_body_location_group(value) in {'upper_extremity', 'lower_extremity'} else 0
        )
        self._add_static_feature('tumor_characteristics', 'extremity_tumor')
        if self.data_type_mapping is not None:
            self.data_type_mapping['extremity_tumor'] = int
        return self

    def process_frozen(self):
        return (self.get_initial_pathology_report_date()
                .get_age()
                .get_initial_size()
                .get_episodes_whoops()
                .get_episodes_systemic()
                .get_systemic_timerange()
                .get_episodes_radiotherapy()
                .get_radiotherapy_timerange()
                .get_merge_first_radio_into_main()
                .get_merge_radiotherapy_indication_raw()
                .get_expand_local_recurrence()
                .get_episodes_local_recurrence()
                .get_local_recurrence_initial_diagnosis_timerange()
                .get_expand_metastasis()
                .get_episodes_metastasis()
                .get_episodes_follow_up()
                .get_merge_grading_biopsy_resection()
                .get_therapy_flag()
                .get_margin_aliases()
                .get_body_location_group()
                .get_extremity_tumor()
                .remove_duplicates_drug_names()
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



