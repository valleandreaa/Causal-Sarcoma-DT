from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
from typing  import Tuple
from etl_sdt.utils.logging_config import logger

class PathologyNERExtractor:
    def __init__(self, model_name: str = "alvaroalon2/biobert_genetic_ner"):
        """
        Initializes the PathologyNERExtractor class.

        Parameters:
        - model_name (str): The name of the pre-trained model to use for NER.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline('ner', model=self.model, tokenizer=self.tokenizer)

    def extract_entities(self, text: str) -> list:
        """
        Extracts named entities from the given text.

        Parameters:
        - text (str): The input text.

        Returns:
        - list: A list of dictionaries representing the extracted entities.
        """
        return self.ner_pipeline(text)

    def _rename_column_ner(self, df: pd.DataFrame, column_name: str) -> Tuple[pd.DataFrame, str]:
        """
        Renames a column in a DataFrame to indicate that it contains NER results.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - column_name (str): The name of the column to rename.

        Returns:
        - Tuple[pd.DataFrame, str]: A tuple containing the renamed DataFrame and the new column name.
        """
        new_column_name = f'{column_name}_ner'
        df[new_column_name] = None
        return df, new_column_name

    def extract_entities_dynamic(self, entities: list) -> dict:
        """
        Extracts dynamic entities from a list of entities.

        Parameters:
        - entities (list): A list of dictionaries representing the entities.

        Returns:
        - dict: A dictionary containing the extracted dynamic entities.
        """
        pathology_info = {}
        current_entity = None
        current_text = []

        for entity in entities:

            if 'entity' in entity:
                entity_type = entity['entity']
                entity_text = entity['word'].replace('▁', ' ').strip()

                if entity_type.startswith('B-'):
                    if current_entity is not None and current_text:
                        category = current_entity.split('-')[-1].lower()
                        if category not in pathology_info:
                            pathology_info[category] = []
                        pathology_info[category].append("".join(current_text).replace('##', ''))

                    current_entity = entity_type
                    current_text = [entity_text]
                elif entity_type.startswith('I-') and current_entity is not None:
                    current_text.append(entity_text)

        if current_entity is not None and current_text:
            category = current_entity.split('-')[-1].lower()
            if category not in pathology_info:
                pathology_info[category] = []
            pathology_info[category].append("".join(current_text).replace('##', ''))

        return pathology_info

    def extract_pathology_info(self, entities: list) -> dict:
        """
        Extracts pathology information from a list of entities.

        Parameters:
        - entities (list): A list of dictionaries representing the entities.

        Returns:
        - dict: A dictionary containing the extracted pathology information.
        """
        pathology_info = {
            "genetic": [],
        }

        current_entity = None
        current_text = []

        for entity in entities:
            entity_type = entity['entity']
            entity_text = entity['word']

            if entity_type.startswith('B-'):
                if current_entity is not None and current_text:
                    category = current_entity.split('-')[-1].lower()
                    pathology_info[category].append("".join(current_text).replace('##', ''))

                current_entity = entity_type
                current_text = [entity_text]
            elif entity_type.startswith('I-') and current_entity is not None:
                current_text.append(entity_text)

        if current_entity is not None and current_text:
            category = current_entity.split('-')[-1].lower()
            pathology_info[category].append("".join(current_text).replace('##', ''))

        return pathology_info

    def unique_annotations(self, pathology_info: dict) -> dict:
        """
        Removes duplicate annotations from the pathology information.

        Parameters:
        - pathology_info (dict): A dictionary containing the pathology information.

        Returns:
        - dict: A dictionary with unique annotations.
        """
        logger.info("Removing duplicate annotations from pathology information")
        for category in pathology_info:
            pathology_info[category] = list(set(pathology_info[category]))
        logger.info("Duplicate removal completed")
        return pathology_info
    
