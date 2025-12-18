#!/usr/bin/env python3
import os
import argparse
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
from typing import Tuple, Optional, List

from Models.utils.helpers import MongoExtractor

from Inference.refined_evidence import filter_ajcc_stage_iii

# ------------------------------------------------------------------ #
# Column constants reused for aggregation grouping (must match Mongo schema)
# ------------------------------------------------------------------ #
LESION_SITE_COL = "tumor_characteristics.lesion_site"
FNCLCC_GRADING_COL = "tumor_characteristics.grading_fnclcc"

# Set of lesion_site values considered "Extremity" (aligned with analysis.py)
EXTREMITY_SITE_SET = {
    "femur", "tibia", "knee", "Acetabulum", "Axilla/scapula"
}

def map_extremity_group(val) -> str:
    """Map raw lesion_site to Extremity / Non-extremity / Unknown.
    Adjust EXTREMITY_SITE_SET if clinical definition changes.
    """
    if pd.isna(val):
        return "Unknown"
    return "Extremity" if str(val) in EXTREMITY_SITE_SET else "Non-extremity"
def load_dataframe(config_file: str) -> pd.DataFrame:
    """Load patient data from MongoDB using MongoExtractor."""
    extractor = MongoExtractor(
        connection_string=os.getenv("MONGO_URI"),
        database_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        config_file=config_file,
    )
    df = extractor.get_dataframe()
    df= filter_ajcc_stage_iii(df)
    # val_patient_ids = [156590, 215635, 242834, 323244, 345717, 526227, 527406, 541208,
    #    543042, 554253, 557218, 596485, 597316, 620309, 623524, 630105,
    #    643779, 644919, 666586, 685885, 722597, 729882, 731534, 737027,
    #    740578, 746581, 754544, 758485, 772731, 786252, 788512, 799525,
    #    806949, 883980, 1004685, 1009921, 1019291, 1024164, 1024170,
    #    1024802, 1027820, 1039395, 1040820, 1041181, 1043175, 1043203,
    #    1045021, 1048981, 1051298, 1059922, 1062642, 1062817, 1066772,
    #    1067388, 1068383, 1068420, 1069056, 1073393, 1075614, 1076361,
    #    1082196, 1083297, 1089543, 1093501, 1108596, 1121411, 1129171,
    #    1129883, 1135797, 1135959, 1137744, 1150095, 1156225, 1163146,
    #    1164292, 1164583, 1166965, 1168780, 1170493, 1172118, 1179808,
    #    1182831, 1196929, 1201406, 1210516, 1218316, 1227669, 1234070,
    #    1367536, 1482920, 1623575, 1864637, 2062965, 2965739, 3009605,
    #    3046004, 3070279, 3132164, 3218244, 3713571, 3931935, 3998282,
    #    4072073, 4133915, 4136202, 4173960, 4258568, 4391969, 4597575,
    #    5124433, 7005393, 7027290, 7041863, 7042017, 7063452, 7077825,
    #    7089896, 7094437, 7095159, 7127671, 7189917, 7193543, 7212730,
    #    7220739, 7224489, 7270384, 7290020, 7302677, 7355593, 7389708,
    #    7390813, 7396520, 7397603, 7398851, 7530399, 8041667, 8043280,
    #    8056530, 8099932, 8113981, 8136245, 8142960, 8243730, 8329971,
    #    8404797, 8502025, 8553685, 8563359, 8574708, 11145886, 11188681,
    #    11199316, 11203520, 100661653, '1051442ksw', '1051612ksw',
    #    '1093406ksw', '1106490ksw', '1124750ksw', '1130473ksw',
    #    '453800ksw'] 
    # # # Filter by validation patient IDs
    # # val_patient_ids = [
    # #     1019562,
    # #     7549036,
    # #     11087934,
    # #     156590,
    # #     11182976,
    # #     807705,
    # #     7047993,
    # #     11235569,
    # #     "608927ksw",
    # #     "1031579ksw",
    # #     11214100,
    # #     100783061,
    # #     643779,
    # #     254801,
    # #     8530105,
    # #     1081132,
    # #     8243730,
    # #     7306607,
    # #     1938649,
    # #     2078700,
    # #     429237,
    # #     "1041910ksw",
    # #     "1043154ksw",
    # #     788512,
    # #     630105,
    # #     378937,
    # #     547718,
    # #     7512785,
    # #     8046982,
    # #     724985,
    # #     11135940,
    # #     11209564,
    # #     3387453,
    # #     11054891,
    # #     7008643,
    # #     11059470,
    # #     1096743,
    # #     8400173,
    # #     "1072741ksw",
    # #     1076998,
    # #     "1159023ksw",
    # #     1103470,
    # #     737027,
    # #     7241196,
    # #     3132164,
    # #     7546042,
    # #     1051298,
    # #     1187592,
    # #     11010372,
    # #     1049321,
    # #     432862,
    # #     1108596,
    # #     3787370,
    # #     1189147,
    # #     "1094152ksw",
    # #     1067388,
    # #     399447,
    # #     1197312,
    # #     1060591,
    # #     8574708,
    # #     1117034,
    # #     1861972,
    # #     7271801,
    # #     1001523,
    # #     "453800ksw",
    # #     8358676,
    # #     1009734,
    # #     11105013,
    # #     468387,
    # #     1106794,
    # #     1046388,
    # #     1242676,
    # #     1367536,
    # #     974226,
    # #     1264843,
    # #     7210343,
    # #     "1029742ksw",
    # #     200683833,
    # #     599774,
    # #     "1084900ksw",
    # #     511225,
    # #     1138511,
    # #     1067138,
    # #     376056,
    # #     457007,
    # #     1035025,
    # #     11032261,
    # #     1257289,
    # #     11140299,
    # #     11169073,
    # #     1150095,
    # #     8523532,
    # #     501398,
    # #     1858971,
    # #     679704,
    # #     8371682,
    # #     7250508,
    # #     365946,
    # #     1102534,
    # #     11096881,
    # #     1229129,
    # #     1252042,
    # #     576627,
    # #     1864637,
    # #     550712,
    # #     "1130473ksw",
    # #     1214822,
    # #     1055539,
    # #     802149,
    # #     "1051612ksw",
    # #     1219380,
    # #     "1101660ksw",
    # #     809812,
    # #     217482,
    # #     1128802,
    # #     11070462,
    # #     7153203,
    # #     1061876,
    # #     4760794,
    # #     "1124750ksw",
    # #     11264743,
    # #     1039395,
    # #     8608321,
    # #     298338,
    # #     11093547,
    # #     8034078,
    # #     1068383,
    # #     1059922,
    # #     554253,
    # #     322863,
    # #     323244,
    # #     5383358,
    # #     11078569,
    # #     527406,
    # #     8484374,
    # #     "1065132ksw",
    # #     610110,
    # #     729747,
    # #     1062817,
    # #     1075614,
    # #     1124790,
    # #     294030,
    # #     633675,
    # #     11191523
    # # ]
    
    # # # Convert patient IDs to handle both integers and strings for comparison
    # if '_id' in df.columns:
    #     # Create both string and numeric versions of validation IDs for robust matching
    #     val_patient_ids_str = [str(pid) for pid in val_patient_ids]
    #     val_patient_ids_num = []
    #     for pid in val_patient_ids:
    #         try:
    #             # Try to convert to int if it's a numeric string
    #             if isinstance(pid, str) and not pid.endswith('ksw'):
    #                 val_patient_ids_num.append(int(pid))
    #             elif isinstance(pid, (int, float)):
    #                 val_patient_ids_num.append(int(pid))
    #         except (ValueError, TypeError):
    #             pass  # Skip non-numeric strings
        
    #     # Filter using both string and numeric versions to catch all matches
    #     original_count = len(df)
    #     df_filtered = df[
    #         df['_id'].isin(val_patient_ids) |  # Original mixed types
    #         df['_id'].isin(val_patient_ids_str) |  # All as strings
    #         df['_id'].isin(val_patient_ids_num)  # All as numbers (excluding 'ksw' strings)
    #     ]
    #     filtered_count = len(df_filtered)
        
    #     print(f"Filtered dataset: {filtered_count} rows from {original_count} total rows")
    #     print(f"Unique patients in filtered data: {df_filtered['_id'].nunique()}")
        
    #     return df_filtered
    # else:
    #     print("Warning: '_id' column not found. Returning unfiltered data.")
    return df



def summarize_numeric(series: pd.Series) -> Tuple[float, float, float, float, float, int, float]:
    """Return mean, median, std, 95% CI, missing count, and missing percentage for a numeric series."""
    clean = series.dropna().astype(float)
    total_count = len(series)
    missing_count = total_count - len(clean)
    missing_percentage = 100 * missing_count / total_count if total_count > 0 else 0.0
    
    if clean.empty:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), missing_count, missing_percentage
    mean = clean.mean()
    median = clean.median()
    std = clean.std()
    n = len(clean)
    margin = 1.96 * std / np.sqrt(n)
    lower = mean - margin
    upper = mean + margin
    return mean, median, std, lower, upper, missing_count, missing_percentage


def summarize_categorical(series: pd.Series) -> str:
    """Summarize categorical data showing all categories."""
    # Check if this is a one-hot encoded feature
    if is_one_hot_encoded(series):
        counts = series.value_counts(dropna=False)
        percentages = 100 * counts / len(series)
        
        parts = []
        # For one-hot encoded, show as No/Yes instead of 0/1
        for idx, cnt, perc in zip(counts.index, counts, percentages):
            if pd.isna(idx):
                label = "NaN"
            elif idx == 0 or idx == 0.0:
                label = "No"
            elif idx == 1 or idx == 1.0:
                label = "Yes"
            else:
                label = str(idx)
            parts.append(f"{label}={perc:.1f}%({cnt})")
    else:
        counts = series.value_counts(dropna=False)
        percentages = 100 * counts / len(series)
        
        parts = []
        for idx, cnt, perc in zip(counts.index, counts, percentages):
            cat_name = str(idx)
            parts.append(f"{cat_name}={perc:.1f}%({cnt})")
    
    # Format with line breaks for better readability if there are many categories
    if len(parts) > 3:
        # Group categories into chunks for better formatting
        result = "; ".join(parts[:3])
        if len(parts) > 3:
            remaining = "; ".join(parts[3:])
            result += ";\n" + remaining
    else:
        result = "; ".join(parts)
    
    return result


def is_one_hot_encoded(series: pd.Series) -> bool:
    """Check if a numeric series is one-hot encoded (only contains 0, 1, and possibly NaN)."""
    unique_values = set(series.dropna().unique())
    return unique_values.issubset({0, 1, 0.0, 1.0})


class TeeStream:
    """Tee stdout/stderr to a file while keeping console output."""
    def __init__(self, file_path: str, also_stderr: bool = True, mode: str = 'w') -> None:
        self.file_path = file_path
        self.also_stderr = also_stderr
        self.mode = mode
        self._file = None
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

    def __enter__(self):
        self._file = open(self.file_path, self.mode, encoding='utf-8')
        sys.stdout = self
        if self.also_stderr:
            sys.stderr = self
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file:
            try:
                self._file.flush()
            finally:
                self._file.close()
        sys.stdout = self._orig_stdout
        if self.also_stderr:
            sys.stderr = self._orig_stderr

    def write(self, data: str) -> None:
        try:
            if self._file:
                self._file.write(data)
        finally:
            if self._orig_stdout:
                self._orig_stdout.write(data)

    def flush(self) -> None:
        try:
            if self._file:
                self._file.flush()
        finally:
            if self._orig_stdout:
                self._orig_stdout.flush()


def aggregate_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data by patient (_id), handling different types of features appropriately.
    
    Aggregation strategy:
    - Numeric features: Use mean across time steps per patient
    - One-hot encoded features: Use max (if any time step is 1, patient has that feature)
    - Categorical features: Use mode (most common value) or last value if no mode
    
    This ensures that statistics represent patient-level characteristics rather than
    time-step-level observations.
    """
    if '_id' not in df.columns:
        raise ValueError("DataFrame must contain '_id' column for patient aggregation")
    
    # Separate columns by type for appropriate aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "_id" in numeric_cols:
        numeric_cols.remove("_id")
    
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if "_id" in categorical_cols:
        categorical_cols.remove("_id")
    
    # Identify one-hot encoded columns among numeric columns
    onehot_cols = []
    truly_numeric_cols = []
    
    treatment_cols = [c for c in df.columns if 'surgery' in c or 'chemotherapy' in c or 'radiotherapy' in c]

    for col in numeric_cols:
        if is_one_hot_encoded(df[col]) or col in treatment_cols:
            onehot_cols.append(col)
        else:
            truly_numeric_cols.append(col)
    
    # Prepare aggregation dictionary
    agg_dict = {}
    
    # For truly numeric columns, use mean (could also consider max, last, etc.)
    for col in truly_numeric_cols:
        agg_dict[col] = 'mean'
    
    # For one-hot encoded columns, use max (if any time step is 1, patient has that feature)
    for col in onehot_cols:
        agg_dict[col] = 'max'
    
    # For categorical columns, use the most common value (mode) or last value
    for col in categorical_cols:
        agg_dict[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[-1]
    
    # Group by patient and aggregate
    patient_df = df.groupby('_id').agg(agg_dict).reset_index()
    
    print(f"Original data: {len(df)} rows")
    print(f"Patient-level data: {len(patient_df)} patients")
    print(f"Average time steps per patient: {len(df) / len(patient_df):.1f}")
    
    return patient_df


def analyze_treatment_combinations(df: pd.DataFrame, level: str = "episode") -> None:
    """Analyze and display frequency of treatment combinations."""
    treatment_cols = ['episodes.surgery', 'episodes.chemotherapy', 'episodes.radiotherapy']
    
    # Check if all treatment columns exist
    missing_cols = [col for col in treatment_cols if col not in df.columns]
    if missing_cols:
        print(f"\nWarning: Treatment columns not found: {missing_cols}")
        return
    
    # Create combinations
    combinations = df[treatment_cols].copy()
    
    # Convert to string representation for better display
    for col in treatment_cols:
        combinations[col] = combinations[col].fillna('NaN').astype(str)
    
    # Count combinations
    combo_counts = combinations.groupby(treatment_cols).size().reset_index(name='count')
    combo_counts = combo_counts.sort_values('count', ascending=False)
    
    # Calculate percentages
    total_count = len(df)
    combo_counts['percentage'] = 100 * combo_counts['count'] / total_count
    
    # Display results
    level_label = "episodes" if level == "episode" else "patients"
    print(f"\nTreatment Combinations ({level} level, N={total_count} {level_label}):")
    print("=" * 100)
    print(f"{'Surgery':<15} {'Chemotherapy':<15} {'Radiotherapy':<15} {'Count':<8} {'Percentage':<12}")
    print("-" * 100)
    
    for _, row in combo_counts.iterrows():
        surgery = row['episodes.surgery']
        chemo = row['episodes.chemotherapy']
        radio = row['episodes.radiotherapy']
        count = row['count']
        pct = row['percentage']
        
        print(f"{surgery:<15} {chemo:<15} {radio:<15} {count:<8} {pct:6.1f}%")
    
    print("=" * 100)
    
    # Additional summary statistics
    print(f"\nTotal unique combinations: {len(combo_counts)}")
    
    # Most common combination
    most_common = combo_counts.iloc[0]
    print(f"Most common combination: Surgery={most_common['episodes.surgery']}, "
          f"Chemo={most_common['episodes.chemotherapy']}, Radio={most_common['episodes.radiotherapy']} "
          f"({most_common['count']} {level_label}, {most_common['percentage']:.1f}%)")


def _bool_from_value(v) -> Optional[int]:
    """Convert common representations to 0/1, return None if unknown."""
    if pd.isna(v):
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    # numbers
    if isinstance(v, (int, np.integer)):
        if v in (0, 1):
            return int(v)
        return 1 if v != 0 else 0
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return None
        return 1 if v != 0.0 else 0
    # strings
    try:
        s = str(v).strip().lower()
    except Exception:
        return None
    if s in {"1", "true", "yes", "y", "t"}:
        return 1
    if s in {"0", "false", "no", "n", "f"}:
        return 0
    # try numeric
    try:
        f = float(s)
        return 1 if f != 0.0 else 0
    except Exception:
        return None


def _derive_treatment_scenario(df: pd.DataFrame,
                               surgery_col: str = 'episodes.surgery',
                               chemo_col: str = 'episodes.chemotherapy',
                               radio_col: str = 'episodes.radiotherapy',
                               out_col: str = 'treatment_scenario') -> pd.DataFrame:
    """Create a scenario label column from surgery/chemo/radiotherapy presence.

    Labels: None, S, C, R, S+C, S+R, C+R, S+C+R, Unknown
    """
    missing = [c for c in [surgery_col, chemo_col, radio_col] if c not in df.columns]
    if missing:
        # If we don't have the columns, do nothing
        print(f"\nWarning: Cannot derive treatment scenario, missing columns: {missing}")
        return df

    s_vals = df[surgery_col].apply(_bool_from_value)
    c_vals = df[chemo_col].apply(_bool_from_value)
    r_vals = df[radio_col].apply(_bool_from_value)

    def label_row(s, c, r) -> str:
        if s is None and c is None and r is None:
            return "Unknown"
        # treat None as 0 for combination presence
        s = 1 if s == 1 else 0
        c = 1 if c == 1 else 0
        r = 1 if r == 1 else 0
        key = (s, c, r)
        mapping = {
            (0, 0, 0): "None",
            (1, 0, 0): "S",
            (0, 1, 0): "C",
            (0, 0, 1): "R",
            (1, 1, 0): "S+C",
            (1, 0, 1): "S+R",
            (0, 1, 1): "C+R",
            (1, 1, 1): "S+C+R",
        }
        return mapping.get(key, "Unknown")

    df[out_col] = [label_row(s, c, r) for s, c, r in zip(s_vals, c_vals, r_vals)]
    return df


def analyze_treatment_scenarios(df: pd.DataFrame, level: str = "episode",
                                histology_col: Optional[str] = None,
                                scenario_col: str = 'treatment_scenario') -> None:
    """Summarize derived treatment scenarios overall and optionally by histology."""
    
    # For episode-level analysis, we need to consider all treatments a patient has had.
    if level == 'episode' and '_id' in df.columns:
        # Create patient-level scenarios first
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        patient_scenarios = _derive_treatment_scenario(df.groupby('_id')[numeric_cols].max().reset_index(), out_col=scenario_col)
        
        # Merge patient-level scenarios back to the episode-level dataframe
        df_with_scenarios = pd.merge(df, patient_scenarios[['_id', scenario_col]], on='_id', how='left')
    else:
        # For patient-level or when no _id, derive scenarios directly
        df_with_scenarios = _derive_treatment_scenario(df, out_col=scenario_col)

    if scenario_col not in df_with_scenarios.columns:
        return

    total = len(df_with_scenarios)
    print(f"\nDerived Treatment Scenarios ({level} level, N={total} {'episodes' if level=='episode' else 'patients'}):")
    print("=" * 100)
    counts = df_with_scenarios[scenario_col].value_counts(dropna=False)
    for label, cnt in counts.items():
        pct = 100.0 * cnt / total if total else 0.0
        print(f"{label:<10} {cnt:>6}  ({pct:5.1f}%)")
    print("=" * 100)

    # Optional by-histology breakdown
    if histology_col and histology_col in df_with_scenarios.columns:
        print(f"\nTreatment Scenarios by Histology ({level} level)")
        print("-" * 100)
        
        # When analyzing by episode, we need to be careful with counting
        if level == 'episode':
            # We want to count patients, not episodes, in the histology breakdown
            patient_level_histology = df_with_scenarios[['_id', histology_col, scenario_col]].drop_duplicates()
            grp = patient_level_histology.groupby(histology_col, dropna=False)[scenario_col].value_counts().rename('count').reset_index()
        else:
            grp = df_with_scenarios.groupby(histology_col, dropna=False)[scenario_col].value_counts().rename('count').reset_index()

        # within-histology percentages
        grp['pct_in_hist'] = grp.groupby(histology_col)['count'].transform(lambda x: 100.0 * x / x.sum())
        # display
        for hist_value, sub in grp.groupby(histology_col, dropna=False):
            hlabel = "NaN" if pd.isna(hist_value) else str(hist_value)
            n_h = int(sub['count'].sum())
            print(f"\n{hlabel} (N={n_h})")
            for _, row in sub.sort_values('count', ascending=False).iterrows():
                print(f"  {row[scenario_col]:<10} {int(row['count']):>6}  ({row['pct_in_hist']:5.1f}%)")


def detect_histology_column(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    """Attempt to detect a histology column in the dataframe.

    Priority:
    1) Explicit preferred column if present.
    2) Any column name containing 'histology' (case-insensitive).
    3) Heuristics: columns containing 'diagnosis' or 'histo' substrings.
    Returns the column name or None if not found.
    """
    if preferred and preferred in df.columns:
        return preferred

    lower_cols = {c.lower(): c for c in df.columns}
    # Direct 'histology' match
    for lc, orig in lower_cols.items():
        if 'histology' in lc:
            return orig
    # Heuristic matches
    for lc, orig in lower_cols.items():
        if 'histo' in lc or 'diagnosis' in lc or 'dx' in lc:
            return orig
    return None


def summarize_histology_distribution(df: pd.DataFrame, histology_col: str) -> None:
    """
    Calculates and prints the frequency and percentage of each histology type.
    """
    if histology_col not in df.columns:
        print(f"\nWarning: Histology column '{histology_col}' not found. Cannot summarize histology distribution.")
        return

    total_patients = len(df)
    histology_counts = df[histology_col].value_counts(dropna=False).reset_index()
    histology_counts.columns = ['Histology', 'Count']
    histology_counts['Percentage'] = (histology_counts['Count'] / total_patients * 100)

    # Handle potential NaN values in histology column for display
    histology_counts['Histology'] = histology_counts['Histology'].fillna('NaN')
    
    # Format percentage
    histology_counts['Percentage'] = histology_counts['Percentage'].apply(lambda x: f"{x:.1f}%")


    print(f"\nHistology Distribution (N={total_patients} patients)")
    print("=" * 80)
    print(tabulate(histology_counts, headers="keys", tablefmt="grid", showindex=False))
    print("=" * 80)


def summarize_by_categorical_group(patient_df: pd.DataFrame, cat_col: str, label_name: str) -> None:
    """Generic patient-level statistics by any categorical grouping column.

    Similar to summarize_by_histology but parameterized. Skips if column missing.
    """
    if cat_col not in patient_df.columns:
        print(f"\nWarning: grouping column '{cat_col}' not found; skipping per-{label_name} stats.")
        return

    # Prepare numeric / one-hot split
    initial_numeric_cols = patient_df.select_dtypes(include=[np.number]).columns.tolist()
    if "_id" in initial_numeric_cols:
        initial_numeric_cols.remove("_id")
    numeric_cols, onehot_cols = [], []
    for col in initial_numeric_cols:
        if is_one_hot_encoded(patient_df[col]):
            onehot_cols.append(col)
        else:
            numeric_cols.append(col)

    cat_cols = patient_df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols.extend(onehot_cols)
    total_patients = len(patient_df)
    print("\n" + "=" * 100)
    print(f"Patient-Level Statistics by {label_name.capitalize()} (N={total_patients} patients)")
    print("=" * 100)
    for gval, gdf in patient_df.groupby(cat_col, dropna=False):
        glabel = "NaN" if pd.isna(gval) else str(gval)
        n = len(gdf)
        pct = 100.0 * n / total_patients if total_patients else 0.0
        print(f"\n--- {label_name.capitalize()}: {glabel} (N={n}, {pct:.1f}%) ---")
        # Numeric
        numeric_summary = []
        for col in numeric_cols:
            mean, median, std, lower, upper, missing_count, missing_pct = summarize_numeric(gdf[col])
            numeric_summary.append([
                col,
                f"{mean:.3f}" if pd.notna(mean) else "nan",
                f"{median:.3f}" if pd.notna(median) else "nan",
                f"{std:.3f}" if pd.notna(std) else "nan",
                f"[{lower:.3f}, {upper:.3f}]" if all(pd.notna([lower, upper])) else "[nan, nan]",
                f"{missing_count} ({missing_pct:.1f}%)",
            ])
        if numeric_summary:
            print("\nNumeric Features:")
            print(tabulate(numeric_summary, headers=["Feature", "Mean", "Median", "Std", "95% CI", "Missing"], tablefmt="grid"))
        # Categorical
        if cat_cols:
            print("\nCategorical Features:")
            for col in cat_cols:
                if col == cat_col:
                    continue
                try:
                    dist = summarize_categorical(gdf[col])
                    print(f"  {col}: {dist}")
                except Exception as e:
                    print(f"  {col}: error summarizing ({e})")
        print("")


def summarize_by_histology(patient_df: pd.DataFrame, histology_col: str) -> None:
    """Print patient-level statistics broken down by histology.

    Uses the same aggregation assumptions as overall stats, but grouped per histology value.
    """
    if histology_col not in patient_df.columns:
        print(f"\nWarning: histology column '{histology_col}' not found; skipping per-histology stats.")
        return

    # Prepare columns similar to overall patient-level analysis
    initial_numeric_cols = patient_df.select_dtypes(include=[np.number]).columns.tolist()
    if "_id" in initial_numeric_cols:
        initial_numeric_cols.remove("_id")

    numeric_cols: List[str] = []
    onehot_cols: List[str] = []
    for col in initial_numeric_cols:
        if is_one_hot_encoded(patient_df[col]):
            onehot_cols.append(col)
        else:
            numeric_cols.append(col)

    # Categorical columns (including one-hot)
    cat_cols = patient_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if histology_col in cat_cols:
        pass
    else:
        # Ensure histology column is treated as categorical for grouping
        cat_cols.append(histology_col)
    cat_cols.extend(onehot_cols)

    total_patients = len(patient_df)
    print("\n" + "=" * 100)
    print(f"Patient-Level Statistics by Histology (N={total_patients} patients)")
    print("=" * 100)

    # Iterate per histology
    groups = patient_df.groupby(histology_col, dropna=False)
    for hist_value, gdf in groups:
        label = "NaN" if pd.isna(hist_value) else str(hist_value)
        n = len(gdf)
        pct = 100.0 * n / total_patients if total_patients else 0.0
        print(f"\n--- Histology: {label} (N={n}, {pct:.1f}%) ---")

        # Numeric summary table
        numeric_summary = []
        for col in numeric_cols:
            mean, median, std, lower, upper, missing_count, missing_pct = summarize_numeric(gdf[col])
            numeric_summary.append(
                [
                    col,
                    f"{mean:.3f}" if pd.notna(mean) else "nan",
                    f"{median:.3f}" if pd.notna(median) else "nan",
                    f"{std:.3f}" if pd.notna(std) else "nan",
                    f"[{lower:.3f}, {upper:.3f}]" if all(pd.notna([lower, upper])) else "[nan, nan]",
                    f"{missing_count} ({missing_pct:.1f}%)",
                ]
            )

        if numeric_summary:
            print("\nNumeric Features:")
            print(tabulate(numeric_summary, headers=["Feature", "Mean", "Median", "Std", "95% CI", "Missing"], tablefmt="grid"))

        # Categorical summary
        if cat_cols:
            print("\nCategorical Features:")
            for col in cat_cols:
                if col == histology_col:
                    # Skip echoing the grouping column's distribution inside its own group
                    continue
                try:
                    dist = summarize_categorical(gdf[col])
                    print(f"  {col}: {dist}")
                except Exception as e:
                    print(f"  {col}: error summarizing ({e})")
        print("")


def main(args: argparse.Namespace) -> None:
    df = load_dataframe(args.config_file)




    if args.output_csv:
        df.to_csv(args.output_csv, index=False)

    # Analyze treatment combinations on the original data (before aggregation)
    print("="*80)
    analyze_treatment_combinations(df, level="episode")
    # Also derived treatment scenario distribution (episode level)
    analyze_treatment_scenarios(df, level="episode", histology_col=detect_histology_column(df, args.histology_col))

    # Aggregate data by patient first
    patient_df = aggregate_by_patient(df)
    
    # Save patient-level data if requested
    if args.output_csv:
        patient_csv = args.output_csv.replace('.csv', '_patient_level.csv')
        patient_df.to_csv(patient_csv, index=False)
        print(f"Patient-level data saved to: {patient_csv}")

    # Get initial column classifications from patient-level data
    initial_numeric_cols = patient_df.select_dtypes(include=[np.number]).columns.tolist()
    if "_id" in initial_numeric_cols:
        initial_numeric_cols.remove("_id")
    
    # Separate one-hot encoded columns from truly numeric columns
    numeric_cols = []
    onehot_cols = []
    
    for col in initial_numeric_cols:
        if is_one_hot_encoded(patient_df[col]):
            onehot_cols.append(col)
        else:
            numeric_cols.append(col)
    
    # Get categorical columns (including one-hot encoded ones)
    cat_cols = patient_df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols.extend(onehot_cols)  # Add one-hot encoded columns to categorical

    # Print information about detected one-hot features
    if onehot_cols:
        print(f"\nDetected {len(onehot_cols)} one-hot encoded features (treated as categorical):")
        for col in onehot_cols:
            print(f"  - {col}")

    numeric_summary = []
    for col in numeric_cols:
        mean, median, std, lower, upper, missing_count, missing_pct = summarize_numeric(patient_df[col])
        numeric_summary.append(
            [col, f"{mean:.3f}", f"{median:.3f}", f"{std:.3f}", f"[{lower:.3f}, {upper:.3f}]", f"{missing_count} ({missing_pct:.1f}%)"]
        )

    cat_summary = []
    for col in cat_cols:
        cat_summary.append([col, summarize_categorical(patient_df[col])])

    if numeric_summary:
        print(f"\nNumeric Features (Patient-Level Statistics, N={len(patient_df)} patients):")
        print("=" * 80)
        print(tabulate(numeric_summary, headers=["Feature", "Mean", "Median", "Std", "95% CI", "Missing"], tablefmt="grid"))

    if cat_summary:
        print(f"\nCategorical Features (Patient-Level Statistics, N={len(patient_df)} patients):")
        print("=" * 100)
        for feature, distribution in cat_summary:
            print(f"\n{feature}:")
            print(f"  {distribution}")
        print("\n" + "=" * 100)

    # Optional: Also show patient-level treatment combinations for comparison
    print("\n" + "="*80)
    analyze_treatment_combinations(patient_df, level="patient")
    # Derived treatment scenarios at patient level
    analyze_treatment_scenarios(patient_df, level="patient", histology_col=detect_histology_column(patient_df, args.histology_col))

    # Histology-level statistics (patient level)
    hist_col = detect_histology_column(patient_df, args.histology_col)
    if hist_col:
        print("\n" + "="*80)
        print(f"Using histology column: {hist_col}")
        summarize_histology_distribution(patient_df, hist_col)
        summarize_by_histology(patient_df, hist_col)
    else:
        if args.histology_col:
            print(f"\nWarning: Could not find histology column '{args.histology_col}'.")
        else:
            print("\nNote: No histology column detected; skipping per-histology stats.")

    # ------------------------------------------------------------------ #
    # Extremity vs Non-extremity aggregated statistics
    # ------------------------------------------------------------------ #
    if LESION_SITE_COL in patient_df.columns:
        patient_df['lesion_site_extremity_group'] = patient_df[LESION_SITE_COL].apply(map_extremity_group)
        summarize_by_categorical_group(patient_df, 'lesion_site_extremity_group', 'lesion site group')
    else:
        print(f"\nNote: Lesion site column '{LESION_SITE_COL}' not found; skipping extremity grouping.")

    # ------------------------------------------------------------------ #
    # FNCLCC grading aggregated statistics
    # ------------------------------------------------------------------ #
    if FNCLCC_GRADING_COL in patient_df.columns:
        summarize_by_categorical_group(patient_df, FNCLCC_GRADING_COL, 'FNCLCC grade')
    else:
        print(f"\nNote: FNCLCC grading column '{FNCLCC_GRADING_COL}' not found; skipping grading stats.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient-level statistics from MongoDB (aggregated by patient _id)")
    parser.add_argument(
        "--config_file",
        type=str,
        default="Models/configs/config_temporal_cycle_gan.yaml",
        help="Path to YAML config used by MongoExtractor",
    )
    parser.add_argument("--output_csv", type=str, help="Optional path to export the raw dataframe to CSV (also creates patient-level CSV)")
    parser.add_argument("--histology_col", type=str, default=None, help="Column name to use as histology for per-histology stats (auto-detected if omitted)")
    parser.add_argument("--output_txt", type=str, default='output.txt', help="Path to save all printed output to a text file")
    args = parser.parse_args()
    if args.output_txt:
        os.makedirs(os.path.dirname(args.output_txt) or '.', exist_ok=True)
        with TeeStream(args.output_txt):
            main(args)
    else:
        main(args)