import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


ETL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ETL_ROOT.parent
if str(ETL_ROOT) not in sys.path:
    sys.path.insert(0, str(ETL_ROOT))

from etl_sdt.configs.config_frozen import (  # noqa: E402
    column_renaming_map,
    data_type_mapping,
    feature_mapping,
    label_dict,
)
from etl_sdt.extract.excel_extractor import (  # noqa: E402
    chunking,
    merge_dataframes_on_id,
    open_excel,
)
from etl_sdt.transform.data_transfomer import (  # noqa: E402
    DictionaryTransformer,
    FeatureExtractor,
)
from etl_sdt.utils.logging_config import logger  # noqa: E402


DEFAULT_FILE_PATH = ETL_ROOT / "data" / "20260703_Daten_SSN_frozen20260303_Adjumed_251202_MDS_vs_5.xlsx"
DEFAULT_COLLECTION = "PatientsBNSTS_20260703"

MAIN_SHEET = "MDS"
RT_SHEET = "MDS_RT_alle"
SYSTEMIC_SHEET = "MDS_SYST"

CURATED_COLUMN_ALIASES = {
    "Alt PID = nicht mappen": "alt_patient_id",
    "First patient contact": "date_first_patient_contact",
    "type_biopsy\nTimo-Daten": "biopsy_type",
    "biopsy_neoadjuvant\nTimo-Daten": "biopsy_neoadjuvant",
    "initialsize_a\nTimo-Data": "tumor_max_size_before_surgery",
    "Anatomic region code grouping (calculated) (nicht mehr aktuell)": "anatomic_region_grouping",
    "date_first_patientcontact\nTimo-Data": "date_first_patient_contact",
    "Index-Surgery: Whoops yes/no": "whoops",
    "Date of Whoops (as index surgery)": "date_whoops",
    "Date of index surgery (no Whoops)": "date_index_surgery",
    "Largest tumor dimension\n(surgery)": "tumor_max_size_before_surgery",
    "whoops_initial_diagnosis_code\nTimo-Data": "whoops_initial_diagnosis_code",
    "whoops_surgeon\nTimo-data": "whoops_surgeon",
    "whoops_reoperation_margin_r\nTimo-data": "whoops_reoperation_margin_r",
    "Kommentar Code\nTimo-data": "comment_code",
    "Size A [mm]": "size_a_mm",
    "Size B [mm]": "size_b_mm",
    "Size C [mm]": "size_c_mm",
    "Reason for Chemotherapy": "systemic_treatment_reason",
    "Indication for Radiotherapy": "radiotherapy_indication_raw",
    "ctcae\nTimo-data": "ctcae",
    "(newest) Reason for CTCAE\nTimo-Data": "ctcae_reason_latest",
    "Site of extrapulmonary metastasis: Reality": "site_extrapulmonary_metastasis",
    "Site of extrapulmonary metastasis: SHAPEHub\nlung, pleura, bone, liver, soft tissue, lymph node, brain, retroperitoneal, intraperitoneal, other": "site_extrapulmonary_metastasis",
    "(W) Other diagnoses?\nTimo-data": "other_diagnoses",
    "(newest) Patient history (clinics, therapy) - latest to newest\nTimo-data": "patient_history_latest",
    "Types of radiotherapy": "radiotherapy_type",
    "PTV volume (cm³)": "radiotherapy_ptv",
    "GTV volume (cm³)": "radiotherapy_gtv",
}


BN_STATIC_FEATURES = {
    "general": [
        "age",
        "date_pathology_report",
    ],
    "tumor_characteristics": [
        "biopsy_grading",
        "who_diagnosis_code",
        "body_location_group",
        "anatomic_region_grouping",
        "extremity_tumor",
        "metastasis_present_at_diagnosis",
        "initial_size",
    ],
    "treatments": [
        "any_whoops",
        "radiotherapy_preoperative",
        "any_surgery",
        "any_radiotherapy",
        "any_chemotherapy",
        "any_local_recurrence",
        "any_metastasis",
        "dod",
        "local_recurrence_initial_diagnosis_timerange",
        "pathologist_margin_judgement",
        "pathologist_margin_judgment",
        "pathologist_margin_judgment_pre_rt",
    ],
}


def curated_column_renaming_map():
    rename_map = copy.deepcopy(column_renaming_map)
    rename_map.update(CURATED_COLUMN_ALIASES)
    return rename_map


def curated_label_dict():
    labels = copy.deepcopy(label_dict)
    labels["who_diagnosis_code"]["mapping"].update({
        "Well differentiated liposarcoma": "liposarcoma",
        "Atypical lipomatous tumor": "liposarcoma",
    })
    labels["anatomic_region_grouping"]["mapping"].update({
        "Subfascial": "deep_soft_tissue",
        "Epifascial": "superficial_soft_tissue",
        "Bone": "bone",
    })
    return labels


def normalize_curated_negative_radiotherapy(df):
    negative_values = {
        "no",
        "[0] no",
        "0 no",
        "no radiotherapy",
        "no therapy",
    }

    for column in ("radiotherapy_indication_raw", "radiotherapy_indication"):
        if column not in df.columns:
            continue

        def normalize_value(value):
            if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                return [normalize_value(item) for item in value]
            if isinstance(value, str) and value.strip().lower() in negative_values:
                return "no radiotherapy"
            return value

        df[column] = df[column].apply(normalize_value)

    return df


def curated_feature_mapping():
    mapping = copy.deepcopy(feature_mapping)
    for group_name, field_names in BN_STATIC_FEATURES.items():
        existing = mapping["static"].setdefault(group_name, [])
        for field_name in field_names:
            if field_name not in existing:
                existing.append(field_name)
    return mapping


def drop_dictionary_rows(df, id_column):
    if id_column not in df.columns:
        return df

    id_values = df[id_column].astype(str).str.strip().str.lower()
    cleaned = df.loc[
        df[id_column].notna()
        & ~id_values.isin({"", "nan", "number", "pid", "patient id", "patient id (pid)"})
    ].copy()
    cleaned[id_column] = cleaned[id_column].astype(str).str.strip()
    return cleaned


def load_curated_workbook(file_path):
    df_main = open_excel(file_path=file_path, sheet_name=MAIN_SHEET)
    df_rt = open_excel(file_path=file_path, sheet_name=RT_SHEET)
    df_syst = open_excel(file_path=file_path, sheet_name=SYSTEMIC_SHEET)

    df_main = drop_dictionary_rows(df_main, "Patient ID (PID)")
    df_rt = drop_dictionary_rows(df_rt, "PID")
    df_syst = drop_dictionary_rows(df_syst, "PID")

    return merge_dataframes_on_id(
        df_main,
        df_rt,
        df_syst,
        key_column="Patient ID (PID)",
    )


def collapse_duplicate_columns(df):
    columns = []
    series_by_column = []
    seen = set()

    for column in df.columns:
        if column in seen:
            continue
        seen.add(column)

        duplicate_block = df.loc[:, df.columns == column]
        if duplicate_block.shape[1] == 1:
            series = duplicate_block.iloc[:, 0]
        else:
            series = duplicate_block.bfill(axis=1).iloc[:, 0]
        columns.append(column)
        series_by_column.append(series)

    collapsed = pd.concat(series_by_column, axis=1)
    collapsed.columns = columns
    return collapsed


def convert_numpy_types(record):
    if isinstance(record, dict):
        return {key: convert_numpy_types(value) for key, value in record.items()}
    if isinstance(record, list):
        return [convert_numpy_types(item) for item in record]
    if isinstance(record, pd.Timestamp):
        return record.to_pydatetime()
    if isinstance(record, np.ndarray):
        return record.tolist()
    if isinstance(record, (np.integer, np.floating)):
        return record.item()
    try:
        if pd.isna(record):
            return None
    except (TypeError, ValueError):
        pass
    return record


def keep_nested(record, spec):
    out = {"_id": record["_id"]}
    for top_key, field_names in spec.items():
        source = record.get(top_key, {})
        if not isinstance(source, dict):
            continue
        selected = {field_name: source.get(field_name) for field_name in field_names}
        out[top_key] = selected
    return out


def project_bn_record(record, source_file):
    projected = keep_nested(record, BN_STATIC_FEATURES)
    projected["source"] = {
        "pipeline": "pipeline_curated_bn_sts",
        "workbook": Path(source_file).name,
    }

    if record.get("episodes"):
        projected["episodes"] = record["episodes"]
    if record.get("diagnosis_treatment_sequence"):
        projected["diagnosis_treatment_sequence"] = record["diagnosis_treatment_sequence"]

    return convert_numpy_types(projected)


def transform_records(file_path, timeline=False):
    df = load_curated_workbook(file_path)
    chunks = chunking(df, num_chunks=50, chunking_variable_id="Patient ID (PID)")
    records = []

    for df_chunk in chunks:
        dict_transf = DictionaryTransformer(curated_column_renaming_map())
        df_chunk = dict_transf.renaming(df_chunk)
        df_chunk = collapse_duplicate_columns(df_chunk)
        df_chunk = dict_transf.transform_nested_values(df_chunk)
        df_chunk = dict_transf.map_columns_with_partial_labels(df_chunk, curated_label_dict())
        df_chunk = dict_transf.transform_dataframe(df_chunk, data_format_dict=data_type_mapping)
        df_chunk = normalize_curated_negative_radiotherapy(df_chunk)

        feature_extractor = FeatureExtractor(
            df_chunk,
            curated_feature_mapping(),
            data_type_mapping,
        )
        df_chunk = feature_extractor.process_frozen()
        df_chunk = feature_extractor.post_process_frozen()

        chunk_records = dict_transf.restructure_to_records(
            df_chunk,
            curated_feature_mapping(),
            key_column="patient_id",
            timeline=timeline,
        )
        records.extend(project_bn_record(record, file_path) for record in chunk_records)

    return records


def upsert_record(record, collection):
    result = collection.update_one(
        {"_id": record["_id"]},
        {"$set": record},
        upsert=True,
    )
    if result.upserted_id:
        logger.info("Inserted new document with _id: %s", result.upserted_id)
    else:
        logger.info("Updated existing document with _id: %s", record["_id"])


def load_mongo_env():
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    load_dotenv(dotenv_path=ETL_ROOT / ".env", override=True)


def get_collection(collection_name, mongo_uri=None, mongo_db=None):
    load_mongo_env()
    mongo_uri = mongo_uri or os.getenv("MONGO_URI")
    mongo_db = mongo_db or os.getenv("MONGO_DB")
    if not mongo_uri or not mongo_db:
        raise RuntimeError("Mongo configuration missing. Set MONGO_URI and MONGO_DB, or pass --mongo_uri and --mongo_db.")

    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    return db[collection_name]


def main(
    file_path=DEFAULT_FILE_PATH,
    collection_name=DEFAULT_COLLECTION,
    preview=False,
    dry_run=False,
    preview_n=5,
    delete_collection=False,
    timeline=False,
    mongo_uri=None,
    mongo_db=None,
):
    records = transform_records(file_path, timeline=timeline)

    logger.info("Transformed %s BN STS patient records.", len(records))
    if preview or dry_run:
        logger.info("Preview of processed BN STS records:")
        for record in records[:preview_n]:
            logger.info(record)
        return records

    collection = get_collection(collection_name, mongo_uri=mongo_uri, mongo_db=mongo_db)
    if delete_collection:
        logger.info("Dropping collection %s before processing...", collection_name)
        collection.drop()

    for record in records:
        upsert_record(record, collection)

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ETL pipeline for the 20260703 curated workbook, limited to Bayesian-network STS features."
    )
    parser.add_argument("--file_path", type=str, default=str(DEFAULT_FILE_PATH), help="Path to the curated Excel file")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Target MongoDB collection")
    parser.add_argument("--mongo_uri", type=str, default=None, help="MongoDB URI override")
    parser.add_argument("--mongo_db", type=str, default=None, help="MongoDB database override")
    parser.add_argument("--preview", action="store_true", help="Preview transformed records without writing to MongoDB")
    parser.add_argument("--dry_run", action="store_true", help="Alias for --preview")
    parser.add_argument("--preview_n", type=int, default=5, help="Number of records to show for preview/dry-run")
    parser.add_argument("--delete_collection", action="store_true", help="Drop the target collection before inserting")
    parser.add_argument("--timeline", action="store_true", help="Use timeline-based episodes")

    args = parser.parse_args()
    main(
        file_path=args.file_path,
        collection_name=args.collection,
        preview=args.preview,
        dry_run=args.dry_run,
        preview_n=args.preview_n,
        delete_collection=args.delete_collection,
        timeline=args.timeline,
        mongo_uri=args.mongo_uri,
        mongo_db=args.mongo_db,
    )
