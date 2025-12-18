import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tabulate import tabulate
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact

from Models.utils.helpers import MongoExtractor
from Models.models.utils import MetadataHandler
from Models.dataset.tabular_dataset import TabularDatasetPID
from Models.models.decoder_embedders import decode_embedding
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    f1_score,
    brier_score_loss,
)

from sklearn.linear_model import LogisticRegression
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# # Import local functions needed
# from .temporal_inference import (
#     compute_survival_columns,
#     cohort_segments,
#     baseline_balance,
#     love_plot,
#     create_consort_flow_diagram,
#     km_and_hazard,
#     survival_rate_at,
#     chemo_rt_interaction,
#     filter_ajcc_stage_iii
# )


def filter_ajcc_stage_iii(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for AJCC Stage III high-risk STS patients according to 8th edition criteria.

    Stage III criteria:
    - IIIA: T2 (>5cm to ≤10cm) AND G2/G3 AND N0 M0
    - IIIB: T3 (>10cm to ≤15cm) OR T4 (>15cm) AND G2/G3 AND N0 M0

    Any high-grade (G2/3) tumour >5cm without nodal/distant spread = Stage III

    Note: Using actual feature names from temporal_inference.yaml configuration
    """
    stage_iii_df = df.copy()

    if "general.ajcc_uicc" in stage_iii_df.columns:
        stage_iii_df = stage_iii_df[
            stage_iii_df["general.ajcc_uicc"].str.contains(
                r"(III|IV)", case=False, na=False
            )
        ]
    else:
        print(
            "Warning: 'general.ajcc_uicc' column not found for nodal status inference"
        )
        # For Stage III, we expect N0 status, so we'll include all patients

    print(f"Final AJCC Stage III cohort: {len(stage_iii_df['_id'].unique())} patients")

    return stage_iii_df


def cohort_segments(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    """Return mutually exclusive treatment arms and their counts for AJCC Stage III patients."""
    # Ensure we're working with Stage III patients
    stage_iii_df = filter_ajcc_stage_iii(df)

    arms = {
        "S": stage_iii_df[
            (stage_iii_df["surgery"] == 1)
            & (stage_iii_df["chemotherapy"] == 0)
            & (stage_iii_df["radiotherapy"] == 0)
        ],
        "S_CT": stage_iii_df[
            (stage_iii_df["surgery"] == 1)
            & (stage_iii_df["chemotherapy"] == 1)
            & (stage_iii_df["radiotherapy"] == 0)
        ],
        "S_RT": stage_iii_df[
            (stage_iii_df["surgery"] == 1)
            & (stage_iii_df["chemotherapy"] == 0)
            & (stage_iii_df["radiotherapy"] == 1)
        ],
        "S_RT_CT": stage_iii_df[
            (stage_iii_df["surgery"] == 1)
            & (stage_iii_df["chemotherapy"] == 1)
            & (stage_iii_df["radiotherapy"] == 1)
        ],
    }
    counts = {k: len(v) for k, v in arms.items()}
    return arms, counts


def baseline_balance(
    df_pair: pd.DataFrame, covars: List[str], treat_col: str
) -> Dict[str, float]:
    """Compute weighted standardized mean differences after IPTW."""
    from sklearn.preprocessing import LabelEncoder

    # Filter covars to only include columns that exist in the dataframe
    available_covars = [col for col in covars if col in df_pair.columns]

    if not available_covars:
        print(
            f"Warning: None of the specified covariates {covars} are available in the dataframe"
        )
        return {}

    # Create a copy and handle missing values
    df_pair = df_pair.copy()

    # Handle missing values and encode categorical variables
    label_encoders = {}
    for col in available_covars:
        if col in df_pair.columns:
            # Handle missing values first
            if df_pair[col].dtype in ["int64", "float64"]:
                # Numeric columns - fill with median
                df_pair[col] = df_pair[col].fillna(df_pair[col].median())
            else:
                # Categorical columns - fill with mode or 'unknown'
                mode_val = (
                    df_pair[col].mode().iloc[0]
                    if not df_pair[col].mode().empty
                    else "unknown"
                )
                df_pair[col] = df_pair[col].fillna(mode_val)

                # Encode categorical variables as numeric
                le = LabelEncoder()
                # Handle any remaining NaN values by converting to string
                df_pair[col] = df_pair[col].astype(str)
                df_pair[col] = le.fit_transform(df_pair[col])
                label_encoders[col] = le

    # Remove rows where treatment column is missing
    df_pair = df_pair.dropna(subset=[treat_col])

    if df_pair.empty:
        print("Warning: No data remaining after handling missing values")
        return {}

    X = df_pair[available_covars]
    y = df_pair[treat_col]

    # Check if we have both treatment groups
    if len(y.unique()) < 2:
        print("Warning: Only one treatment group present in the data")
        return {}

    # Check if we have enough samples
    if len(X) < 10:
        print(
            f"Warning: Only {len(X)} samples available, may not be sufficient for reliable propensity score estimation"
        )

    # Ensure all data is numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)  # Fill any remaining NaN with 0

    try:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        ps = model.predict_proba(X)[:, 1]

        # Clip propensity scores to avoid extreme weights
        ps = np.clip(ps, 0.01, 0.99)
        weights = np.where(y == 1, 1.0 / ps, 1.0 / (1.0 - ps))

        # Clip weights to avoid extreme values
        weights = np.clip(weights, 0.1, 10.0)

        df_pair["w"] = weights
        smd: Dict[str, float] = {}

        for col in available_covars:
            try:
                t = df_pair[df_pair[treat_col] == 1]
                c = df_pair[df_pair[treat_col] == 0]

                if len(t) == 0 or len(c) == 0:
                    smd[col] = 0.0
                    continue

                t_mean = np.average(t[col], weights=t["w"])
                c_mean = np.average(c[col], weights=c["w"])
                t_var = np.average((t[col] - t_mean) ** 2, weights=t["w"])
                c_var = np.average((c[col] - c_mean) ** 2, weights=c["w"])
                pooled = np.sqrt((t_var + c_var) / 2)
                smd[col] = (
                    float(np.abs(t_mean - c_mean) / pooled) if pooled > 0 else 0.0
                )
            except Exception as e:
                print(f"Warning: Could not compute SMD for {col}: {e}")
                smd[col] = 0.0

        return smd

    except Exception as e:
        print(f"Error in propensity score estimation: {e}")
        return {}


def love_plot(smd: Dict[str, float], filename: str) -> None:
    """Create a love plot of standardized mean differences with AJCC Stage III context."""
    try:
        if not smd:
            print(f"No SMD data available for love plot: {filename}")
            return

        plt.figure(figsize=(8, max(len(smd) * 0.4, 3)))
        keys = list(smd.keys())
        vals = [smd[k] for k in keys]

        # Color bars based on balance threshold
        colors = ["red" if v > 0.1 else "green" for v in vals]

        bars = plt.barh(keys, vals, color=colors, alpha=0.7)
        plt.axvline(
            0.1, color="red", linestyle="--", alpha=0.8, label="0.1 SMD threshold"
        )
        plt.axvline(
            0.05, color="orange", linestyle=":", alpha=0.6, label="0.05 SMD threshold"
        )

        plt.xlabel("Standardized Mean Difference")
        plt.title(
            f"AJCC Stage III STS: Baseline Balance Assessment\n{filename.replace('_', ' ').replace('.png', '').title()}"
        )
        plt.legend()

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            plt.text(
                val + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Love plot saved: {filename}")

    except Exception as exc:  # pragma: no cover - plotting optional
        print(f"Love plot failed for {filename}: {exc}")


def create_consort_flow_diagram(
    df: pd.DataFrame,
    stage_iii_df: pd.DataFrame,
    arms: Dict[str, pd.DataFrame],
    filename: str = "consort_flow_stage_iii.png",
) -> None:
    """Create a CONSORT-style flow diagram for AJCC Stage III patient allocation."""
    try:
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")

        # Colors
        inclusion_color = "#E8F4FD"
        exclusion_color = "#FFF2CC"
        allocation_color = "#D5E8D4"

        # Title
        ax.text(
            5,
            11.5,
            "AJCC Stage III High-Risk Soft Tissue Sarcoma\nCohort Allocation Flow",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        # Initial cohort
        total_patients = len(df)
        rect1 = patches.FancyBboxPatch(
            (2, 10),
            6,
            1,
            boxstyle="round,pad=0.1",
            facecolor=inclusion_color,
            edgecolor="black",
        )
        ax.add_patch(rect1)
        ax.text(
            5,
            10.5,
            f"Total Patients in Database\n(N = {total_patients})",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        # Arrow down
        ax.arrow(
            5, 9.8, 0, -0.4, head_width=0.1, head_length=0.1, fc="black", ec="black"
        )

        # Stage III inclusion
        stage_iii_n = len(stage_iii_df)
        excluded_n = total_patients - stage_iii_n

        rect2 = patches.FancyBboxPatch(
            (2, 8.5),
            6,
            1,
            boxstyle="round,pad=0.1",
            facecolor=inclusion_color,
            edgecolor="black",
        )
        ax.add_patch(rect2)
        ax.text(
            5,
            9,
            f"AJCC Stage III Patients\n(N = {stage_iii_n})\n• Size >5cm • Grade G2/G3 • N0 M0",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        # Exclusion box
        if excluded_n > 0:
            rect_ex = patches.FancyBboxPatch(
                (8.2, 8.5),
                1.7,
                1,
                boxstyle="round,pad=0.05",
                facecolor=exclusion_color,
                edgecolor="red",
            )
            ax.add_patch(rect_ex)
            ax.text(
                9.05,
                9,
                f"Excluded\n(N = {excluded_n})",
                ha="center",
                va="center",
                fontsize=8,
            )
            ax.arrow(
                8, 9, 0.15, 0, head_width=0.05, head_length=0.05, fc="red", ec="red"
            )

        # Arrow down to allocation
        ax.arrow(
            5, 8.3, 0, -0.8, head_width=0.1, head_length=0.1, fc="black", ec="black"
        )

        # Allocation header
        ax.text(
            5,
            7.2,
            "Treatment Allocation",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        # Treatment arms
        arm_positions = [(1.5, 5.5), (3.5, 5.5), (5.5, 5.5), (7.5, 5.5)]
        arm_names = ["S", "S_CT", "S_RT", "S_RT_CT"]
        arm_labels = [
            "Surgery Only",
            "Surgery +\nChemotherapy",
            "Surgery +\nRadiotherapy",
            "Surgery + RT +\nChemotherapy",
        ]

        for i, (pos, name, label) in enumerate(
            zip(arm_positions, arm_names, arm_labels)
        ):
            count = len(arms[name])
            rect = patches.FancyBboxPatch(
                (pos[0] - 0.7, pos[1] - 0.7),
                1.4,
                1.4,
                boxstyle="round,pad=0.1",
                facecolor=allocation_color,
                edgecolor="black",
            )
            ax.add_patch(rect)
            ax.text(
                pos[0],
                pos[1],
                f"{label}\n(N = {count})",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

            # Arrows from allocation to arms
            ax.arrow(
                5,
                6.8,
                pos[0] - 5,
                pos[1] - 6.8 + 0.7,
                head_width=0.05,
                head_length=0.05,
                fc="black",
                ec="black",
                alpha=0.7,
            )

        # Comparison pairs
        ax.text(
            2.5,
            4,
            "Comparison 1:\nS vs S + CT\nfor incremental\nchemotherapy effect",
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )

        ax.text(
            6.5,
            4,
            "Comparison 2:\nS + RT vs S + RT + CT\nfor incremental\nchemotherapy effect",
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        )

        # Arrows to comparisons
        ax.arrow(
            2.2, 4.8, 0.3, -0.5, head_width=0.05, head_length=0.05, fc="blue", ec="blue"
        )
        ax.arrow(
            2.8,
            4.8,
            -0.3,
            -0.5,
            head_width=0.05,
            head_length=0.05,
            fc="blue",
            ec="blue",
        )
        ax.arrow(
            6.2,
            4.8,
            0.3,
            -0.5,
            head_width=0.05,
            head_length=0.05,
            fc="green",
            ec="green",
        )
        ax.arrow(
            6.8,
            4.8,
            -0.3,
            -0.5,
            head_width=0.05,
            head_length=0.05,
            fc="green",
            ec="green",
        )

        # Study focus
        ax.text(
            5,
            2.5,
            "Primary Research Question:\nDoes perioperative chemotherapy improve outcomes\nwhen added to surgery-based treatment in AJCC Stage III STS?",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"CONSORT flow diagram saved: {filename}")

    except Exception as exc:
        print(f"CONSORT diagram creation failed: {exc}")


def km_and_hazard(
    df_pair: pd.DataFrame, time_col: str, event_col: str, treat_col: str, prefix: str
) -> Dict[str, Any]:
    """Plot KM curves and compute hazard ratio for a treatment pair in AJCC Stage III patients."""
    km = KaplanMeierFitter()
    metrics: Dict[str, Any] = {}

    if df_pair.empty:
        print(f"No data available for {prefix} analysis")
        return metrics

    # Check if required columns exist, if not try to compute them
    if time_col not in df_pair.columns or event_col not in df_pair.columns:
        print(
            f"Computing survival columns for {prefix} as {time_col} and {event_col} not found"
        )
        df_pair = compute_survival_columns(df_pair)

    # Check if we still don't have the required columns
    if time_col not in df_pair.columns or event_col not in df_pair.columns:
        print(
            f"Cannot perform survival analysis for {prefix}: missing {time_col} or {event_col}"
        )
        return metrics

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        treatment_labels = {0: "Control", 1: "Treatment"}
        colors = ["blue", "red"]

        for idx, (val, lbl) in enumerate([(0, "Control"), (1, "Treatment")]):
            subset = df_pair[df_pair[treat_col] == val]
            if len(subset) > 0:
                # Remove rows with missing survival data
                subset_clean = subset[[time_col, event_col]].dropna()
                if len(subset_clean) > 0:
                    km.fit(
                        subset_clean[time_col],
                        subset_clean[event_col],
                        label=f"{lbl} (n={len(subset_clean)})",
                    )
                    km.plot(ax=ax, color=colors[idx], ci_show=True)

                    # Store individual survival curves data
                    metrics[f"survival_function_{lbl.lower()}"] = (
                        km.survival_function_.copy()
                    )
                else:
                    print(f"No valid survival data for {lbl} group in {prefix}")

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Survival Probability")
        ax.set_title(
            f"AJCC Stage III STS: Kaplan-Meier Curves\n{prefix.replace('_', ' ').title()}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{prefix}_km_stage_iii.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Kaplan-Meier plot saved: {prefix}_km_stage_iii.png")

        # Cox proportional hazards model
        cox_data = df_pair[[time_col, event_col, treat_col]].dropna()
        if len(cox_data) > 10 and len(cox_data[treat_col].unique()) == 2:
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col=time_col, event_col=event_col)

            # Extract hazard ratio and confidence interval
            hr = float(np.exp(cph.params_[treat_col]))
            ci = cph.confidence_intervals_.loc[treat_col].to_numpy()
            p_value = float(cph.summary.loc[treat_col, "p"])

            metrics["hazard_ratio"] = hr
            metrics["hr_ci_lower"] = float(np.exp(ci[0]))
            metrics["hr_ci_upper"] = float(np.exp(ci[1]))
            metrics["hr_p_value"] = p_value
            metrics["hr_significant"] = p_value < 0.05

            print(
                f"  Hazard Ratio: {hr:.3f} (95% CI: {metrics['hr_ci_lower']:.3f}-{metrics['hr_ci_upper']:.3f})"
            )
            print(
                f"  p-value: {p_value:.3f} ({'Significant' if p_value < 0.05 else 'Not significant'})"
            )

            # Log-rank test p-value
            metrics["logrank_p"] = p_value

        else:
            print(
                f"Insufficient data for Cox regression in {prefix} (n={len(cox_data)})"
            )

    except Exception as exc:  # pragma: no cover - optional
        print(f"KM/hazard computation failed for {prefix}: {exc}")

    return metrics


def compute_survival_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute survival time and event columns from available date and status fields."""
    from datetime import datetime

    df = df.copy()

    # Find date columns dynamically
    date_diagnosis_col = None
    date_follow_up_col = None

    for col in df.columns:
        if "date_of_diagnosis" in col.lower():
            date_diagnosis_col = col
        elif "date_follow_up" in col.lower():
            date_follow_up_col = col

    # Try to compute survival time from dates
    if date_diagnosis_col and date_follow_up_col:
        try:
            survival_times = []
            successful_conversions = 0

            for _, row in df.iterrows():
                date_diagnosis = row.get(date_diagnosis_col)
                date_follow_up = row.get(date_follow_up_col)

                try:
                    # Parse diagnosis date
                    if isinstance(date_diagnosis, str):
                        diagnosis_date = datetime.fromisoformat(
                            date_diagnosis.replace("Z", "+00:00")
                        )
                    elif isinstance(date_diagnosis, dict) and "$date" in date_diagnosis:
                        diagnosis_date = datetime.fromisoformat(
                            date_diagnosis["$date"].replace("Z", "+00:00")
                        )
                    elif hasattr(
                        date_diagnosis, "to_pydatetime"
                    ):  # handles pd.Timestamp and similar types
                        diagnosis_date = date_diagnosis.to_pydatetime()
                    else:
                        survival_times.append(5.0)  # Default
                        continue

                    # Parse follow-up date
                    if isinstance(date_follow_up, str):
                        follow_up_date = datetime.fromisoformat(
                            date_follow_up.replace("Z", "+00:00")
                        )
                    elif isinstance(date_follow_up, dict) and "$date" in date_follow_up:
                        follow_up_date = datetime.fromisoformat(
                            date_follow_up["$date"].replace("Z", "+00:00")
                        )
                    elif hasattr(
                        date_follow_up, "to_pydatetime"
                    ):  # handles pd.Timestamp and similar types
                        follow_up_date = date_follow_up.to_pydatetime()
                    else:
                        survival_times.append(5.0)  # Default
                        continue

                    # Calculate time difference in years
                    time_diff_years = (follow_up_date - diagnosis_date).days / 365.25
                    survival_times.append(time_diff_years)
                    successful_conversions += 1

                except (ValueError, TypeError):
                    survival_times.append(5.0)  # Default for failed conversions

            df["time_difference_years"] = survival_times
            print(f"Computed survival time for {successful_conversions} patients")

        except Exception as e:
            print(f"Warning: Could not compute survival time from dates: {e}")
            # Set default survival time if computation fails
            df["time_difference_years"] = 5.0  # Default 5 years
    else:
        print("Warning: Date columns not available, using default survival time")
        df["time_difference_years"] = 5.0  # Default 5 years

    # Try to compute event status from patient status
    if "general.status" in df.columns:
        try:
            # Assume status contains information about whether patient died (event=1) or alive (event=0)
            # This may need adjustment based on actual values in the data
            status_values = df["general.status"].unique()
            print(f"Available status values: {status_values}")

            # Common mappings - adjust as needed based on actual data
            if any(
                val in str(status_values).lower() for val in ["dead", "died", "death"]
            ):
                df["event"] = (
                    df["general.status"]
                    .str.contains("dead|died|death", case=False, na=False)
                    .astype(int)
                )
            elif any(val in str(status_values).lower() for val in ["alive", "living"]):
                df["event"] = (
                    ~df["general.status"].str.contains(
                        "alive|living", case=False, na=True
                    )
                ).astype(int)
            else:
                # Try numeric interpretation (1 = event, 0 = censored)
                df["event"] = (
                    pd.to_numeric(df["general.status"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )

            print(
                f"Computed event status: {df['event'].sum()} events, {len(df) - df['event'].sum()} censored"
            )
        except Exception as e:
            print(f"Warning: Could not compute event status from general.status: {e}")
            df["event"] = 0  # Default to censored
    elif "outcomes.status_last_patient_contact_information" in df.columns:
        try:
            # Alternative outcome field
            outcome_values = df[
                "outcomes.status_last_patient_contact_information"
            ].unique()
            print(f"Available outcome values: {outcome_values}")
            df["event"] = (
                df["outcomes.status_last_patient_contact_information"]
                .str.contains("dead|died|death|deceased", case=False, na=False)
                .astype(int)
            )
            print(
                f"Computed event status from outcomes: {df['event'].sum()} events, {len(df) - df['event'].sum()} censored"
            )
        except Exception as e:
            print(f"Warning: Could not compute event status from outcomes: {e}")
            df["event"] = 0  # Default to censored
    else:
        print("Warning: No status columns available, assuming all patients censored")
        df["event"] = 0  # Default to censored

    return df


def chemo_rt_interaction(
    df: pd.DataFrame, time_col: str, event_col: str
) -> Dict[str, Any]:
    """Fit Cox model with CT, RT and interaction."""
    # Check if required columns exist, if not try to compute them
    if time_col not in df.columns or event_col not in df.columns:
        print(f"Computing survival columns as {time_col} and {event_col} not found")
        df = compute_survival_columns(df)

    # Check if treatment columns exist
    required_cols = [
        time_col,
        event_col,
        "episodes.chemotherapy",
        "episodes.radiotherapy",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Missing columns for interaction analysis: {missing_cols}")
        return {}

    cph = CoxPHFitter()
    df_int = df[required_cols].copy()
    df_int["inter"] = df_int["episodes.chemotherapy"] * df_int["episodes.radiotherapy"]

    # Remove rows with missing values
    df_int = df_int.dropna()

    if len(df_int) < 10:
        print(f"Insufficient data for interaction analysis (n={len(df_int)})")
        return {}

    try:
        cph.fit(df_int, duration_col=time_col, event_col=event_col)
        hr = float(np.exp(cph.params_["inter"]))
        p_val = float(cph.summary.loc["inter", "p"])
        return {"interaction_hr": hr, "interaction_p": p_val}
    except Exception as exc:  # pragma: no cover - optional
        print(f"Interaction test failed: {exc}")
        return {}


def validate_model(
    gx,
    meta: Dict[str, Any],
    extractor: MongoExtractor,
    meta_clin: MetadataHandler,
    meta_treat: MetadataHandler,
    decoder_treat,
    feat_spec,
    oh_map,
    ord_map,
    num_ranges,
    device: torch.device,
) -> Dict[str, Any]:
    """Run validation over patient ids stored in metadata and save comparison data to CSV."""
    val_ids = meta.get("val_patient_ids")
    if not val_ids:
        print("No validation ids found in metadata")
        return {}

    print(f"Running validation on {len(val_ids)} patients...")

    real_decoded: List[pd.DataFrame] = []
    gen_decoded: List[pd.DataFrame] = []
    validation_data = []

    for i, pid in enumerate(val_ids):
        try:
            print(f"Processing validation patient {i+1}/{len(val_ids)}: {pid}")
            extractor = MongoExtractor(
                connection_string=os.getenv("MONGO_URI"),
                database_name=os.getenv("MONGO_DB"),
                collection_name=os.getenv("MONGO_COLLECTION"),
                config_file="Models/configs/temporal_inference.yaml",
            )
            # Get patient data
            patient_df = extractor.get_dataframe()
            if patient_df.empty:
                print(f"No data found for patient {pid}")
                continue

            dataset = TabularDatasetPID(patient_df, meta_clin, meta_treat, seq_len=None)
            loader = DataLoader(dataset, batch_size=48, shuffle=False)
            treat_real, clin_emb, actual_treat = next(iter(loader))

            clin_emb = clin_emb.to(device)
            actual_treat = actual_treat.to(device).float()
            treat_real = treat_real.to(device)

            with torch.no_grad():
                fake = gx(clin_emb, actual_treat)

            # Process each time step
            for step_idx, (t_real, t_fake) in enumerate(zip(treat_real[0], fake[0])):
                try:
                    real_df = decode_embedding(
                        decoder_treat(t_real), feat_spec, oh_map, ord_map, num_ranges
                    )
                    gen_df = decode_embedding(
                        decoder_treat(t_fake),
                        feat_spec,
                        oh_map,
                        ord_map,
                        num_ranges,
                        one_hot_argmax=False,
                    )

                    if not real_df.empty and not gen_df.empty:
                        real_decoded.append(real_df)
                        gen_decoded.append(gen_df)

                        # Store data for CSV export
                        real_dict = real_df.iloc[0].to_dict()
                        gen_dict = gen_df.iloc[0].to_dict()

                        # Create comparison record
                        comparison_record = {
                            "patient_id": pid,
                            "time_step": step_idx,
                            "data_type": "comparison",
                        }

                        # Get ALL unique columns from both real and generated data
                        # This ensures we capture variables that may exist in generated data but not real data
                        all_columns = set(real_dict.keys()) | set(gen_dict.keys())

                        # Add real and generated values for ALL columns
                        for col in all_columns:
                            real_val = real_dict.get(col, None)
                            gen_val = gen_dict.get(col, None)

                            comparison_record[f"real_{col}"] = real_val
                            comparison_record[f"generated_{col}"] = gen_val

                            # Calculate absolute error for numeric columns (only if both values exist and are numeric)
                            if real_val is not None and gen_val is not None:
                                try:
                                    # Check if both values are numeric
                                    real_numeric = pd.to_numeric(
                                        real_val, errors="coerce"
                                    )
                                    gen_numeric = pd.to_numeric(
                                        gen_val, errors="coerce"
                                    )

                                    if pd.notna(real_numeric) and pd.notna(gen_numeric):
                                        comparison_record[f"abs_error_{col}"] = abs(
                                            float(real_numeric) - float(gen_numeric)
                                        )
                                    else:
                                        comparison_record[f"abs_error_{col}"] = None
                                except (ValueError, TypeError):
                                    comparison_record[f"abs_error_{col}"] = None
                            else:
                                comparison_record[f"abs_error_{col}"] = None

                        validation_data.append(comparison_record)

                except Exception as e:
                    print(f"Error processing step {step_idx} for patient {pid}: {e}")
                    continue

        except Exception as e:
            print(f"Error processing patient {pid}: {e}")
            continue

    # Save validation comparison data to CSV
    if validation_data:
        validation_df = pd.DataFrame(validation_data)
        csv_filename = "validation_real_vs_generated_comparison.csv"
        validation_df.to_csv(csv_filename, index=False)
        print(f"Saved validation comparison data to {csv_filename}")
        print(f"CSV contains {len(validation_df)} records from {len(val_ids)} patients")

        # Display summary statistics
        numeric_cols = [
            col for col in validation_df.columns if col.startswith("abs_error_")
        ]
        real_cols = [col for col in validation_df.columns if col.startswith("real_")]
        gen_cols = [
            col for col in validation_df.columns if col.startswith("generated_")
        ]

        # Check for variables that exist in generated but not real data
        real_var_names = {col.replace("real_", "") for col in real_cols}
        gen_var_names = {col.replace("generated_", "") for col in gen_cols}

        generated_only_vars = gen_var_names - real_var_names
        real_only_vars = real_var_names - gen_var_names

        print(f"\nVariable Coverage Analysis:")
        print(
            f"  Variables in both real and generated: {len(real_var_names & gen_var_names)}"
        )
        print(f"  Variables only in generated data: {len(generated_only_vars)}")
        print(f"  Variables only in real data: {len(real_only_vars)}")

        if generated_only_vars:
            print(
                f"  Generated-only variables: {', '.join(sorted(generated_only_vars))}"
            )
        if real_only_vars:
            print(f"  Real-only variables: {', '.join(sorted(real_only_vars))}")

        if numeric_cols:
            print("\nValidation Error Summary:")
            for col in numeric_cols:
                mean_error = validation_df[col].mean()
                std_error = validation_df[col].std()
                print(
                    f"  {col.replace('abs_error_', '')}: MAE = {mean_error:.4f} ± {std_error:.4f}"
                )

    # Compute overall metrics
    metrics = compute_pointwise_metrics(real_decoded, gen_decoded)

    # Add CSV export info to metrics
    metrics["validation_csv_saved"] = len(validation_data) > 0
    metrics["validation_csv_filename"] = (
        "validation_real_vs_generated_comparison.csv" if validation_data else None
    )
    metrics["validation_records"] = len(validation_data)
    metrics["validation_patients"] = len(val_ids)

    # Add variable coverage information to metrics
    if validation_data:
        validation_df = pd.DataFrame(validation_data)
        real_cols = [col for col in validation_df.columns if col.startswith("real_")]
        gen_cols = [
            col for col in validation_df.columns if col.startswith("generated_")
        ]

        real_var_names = {col.replace("real_", "") for col in real_cols}
        gen_var_names = {col.replace("generated_", "") for col in gen_cols}

        metrics["variables_in_both"] = len(real_var_names & gen_var_names)
        metrics["variables_generated_only"] = len(gen_var_names - real_var_names)
        metrics["variables_real_only"] = len(real_var_names - gen_var_names)
        metrics["generated_only_variable_names"] = list(gen_var_names - real_var_names)
        metrics["real_only_variable_names"] = list(real_var_names - gen_var_names)

    return metrics


def compute_pointwise_metrics(
    real: List[pd.DataFrame], generated: List[pd.DataFrame]
) -> Dict[str, Any]:
    """Compute MAE for numeric columns and accuracy/F1 for binary events."""
    metrics: Dict[str, Any] = {}
    if not real or not generated:
        return metrics

    num_cols = [c for c in real[0].columns if pd.api.types.is_numeric_dtype(real[0][c])]
    bin_cols = [
        c for c in real[0].columns if set(real[0][c].dropna().unique()).issubset({0, 1})
    ]

    if num_cols:
        real_num = np.concatenate([df[num_cols].to_numpy() for df in real])
        gen_num = np.concatenate([df[num_cols].to_numpy() for df in generated])
        metrics["mae"] = mean_absolute_error(real_num, gen_num)

    if bin_cols:
        real_bin = np.concatenate([df[bin_cols].to_numpy() for df in real]).flatten()
        gen_bin = (
            np.rint(np.concatenate([df[bin_cols].to_numpy() for df in generated]))
            .astype(int)
            .flatten()
        )
        metrics["accuracy"] = accuracy_score(real_bin, gen_bin)
        metrics["f1"] = f1_score(real_bin, gen_bin, zero_division=0)

    return metrics


def plot_kaplan_meier(
    real_times: np.ndarray,
    real_events: np.ndarray,
    gen_times: np.ndarray,
    gen_events: np.ndarray,
    save_path: None,
) -> None:
    """Plot Kaplan-Meier curves for real and generated cohorts."""
    try:
        from lifelines import KaplanMeierFitter
        import matplotlib.pyplot as plt

        km_real = KaplanMeierFitter()
        km_gen = KaplanMeierFitter()
        km_real.fit(real_times, real_events, label="Real")
        km_gen.fit(gen_times, gen_events, label="Generated")
        ax = km_real.plot()
        km_gen.plot(ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival")
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    except Exception as exc:  # pragma: no cover - plotting optional
        print(f"KM plot failed: {exc}")


def compute_survival_metrics(
    times: np.ndarray, events: np.ndarray, pred_times: np.ndarray
) -> Dict[str, Any]:
    """Compute C-index and Brier score for survival predictions."""
    metrics: Dict[str, Any] = {}
    try:
        # Concordance index using standard definition
        n = len(times)
        assert len(pred_times) == n
        concordant = permissible = 0
        for i in range(n):
            for j in range(i + 1, n):
                if events[i] == events[j] == 0:
                    continue
                if times[i] == times[j]:
                    continue
                permissible += 1
                if (pred_times[i] < pred_times[j] and times[i] < times[j]) or (
                    pred_times[i] > pred_times[j] and times[i] > times[j]
                ):
                    concordant += 1
        metrics["c_index"] = (
            concordant / permissible if permissible > 0 else float("nan")
        )
    except Exception:
        metrics["c_index"] = float("nan")

    try:
        metrics["brier"] = brier_score_loss(events, pred_times)
    except Exception:
        metrics["brier"] = float("nan")
    return metrics


def survival_rate_at(threshold: float, times: np.ndarray, events: np.ndarray) -> float:
    """Return survival rate at a given time threshold."""
    survived = (times >= threshold) | (events == 0)
    return float(np.mean(survived))


def estimate_ate(treated: np.ndarray, control: np.ndarray) -> float:
    """Estimate average treatment effect as difference in mean survival time."""
    return float(np.mean(treated) - np.mean(control))


def plot_sarculator_by_scenario(df: pd.DataFrame, filename: str) -> None:
    """Create a plot comparing sarculator scores across treatment scenarios."""
    if "treatment_scenario" not in df.columns:
        print("Cannot create sarculator plot: missing treatment_scenario column")
        return

    try:
        # Check if either sarculator column exists
        sarculator_cols = [
            col
            for col in ["general.sarculator_five", "general.sarculator_ten"]
            if col in df.columns
        ]

        if not sarculator_cols:
            print("Cannot create sarculator plot: no sarculator score columns found")
            return

        fig, axes = plt.subplots(1, len(sarculator_cols), figsize=(12, 6), sharey=True)
        if len(sarculator_cols) == 1:
            axes = [axes]  # Make it iterable for single plot case

        treatment_scenarios = df["treatment_scenario"].unique()
        colors = ["blue", "green", "orange", "red"]

        for i, col in enumerate(sarculator_cols):
            ax = axes[i]

            # Calculate means for bar chart
            means = []
            stds = []
            ns = []
            labels = []

            for j, scenario in enumerate(sorted(treatment_scenarios)):
                scenario_data = df[
                    (df["treatment_scenario"] == scenario) & df[col].notna()
                ]
                if len(scenario_data) > 0:
                    means.append(scenario_data[col].mean())
                    stds.append(scenario_data[col].std())
                    ns.append(len(scenario_data))
                    labels.append(f"{scenario}\n(n={len(scenario_data)})")

            # Create bar chart with error bars
            bars = ax.bar(
                labels, means, yerr=stds, color=colors[: len(means)], alpha=0.7
            )

            # Add value labels on top of bars
            for bar, mean, std, n in zip(bars, means, stds, ns):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{mean:.2f}±{std:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Set labels and title
            ax.set_xlabel("Treatment Scenario")
            ax.set_title(f'{col.replace("general.", "").replace("_", " ").title()}')
            ax.set_ylim(0, 1.0)  # Sarculator scores are probabilities
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            "Sarculator Scores by Treatment Scenario\nAJCC Stage III STS", fontsize=14
        )
        fig.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Sarculator comparison plot saved: {filename}")

    except Exception as exc:
        print(f"Sarculator plot creation failed: {exc}")


def generate_refined_evidence_package(
    stage_iii_df: pd.DataFrame,
    extractor: MongoExtractor = None,
    meta_clin: MetadataHandler = None,
    meta_treat: MetadataHandler = None,
    gx=None,
    decoder_treat=None,
    feat_spec=None,
    oh_map=None,
    ord_map=None,
    num_ranges=None,
    device: torch.device = None,
    print_probabilities: bool = False,
) -> Dict[str, Any]:
    """Generate comprehensive evidence package for AJCC Stage III high-risk STS patients.

    Implements refined analysis using model inference to prove/refute that peri-operative
    chemotherapy improves outcomes when added to:
    A. surgery alone (S → S + CT)
    B. surgery + radiotherapy (S + RT → S + RT + CT)

    Args:
        stage_iii_df: DataFrame with Stage III patients
        extractor: MongoExtractor for patient data
        meta_clin: Clinical metadata handler
        meta_treat: Treatment metadata handler
        gx: Generator model
        decoder_treat: Treatment decoder
        feat_spec: Feature specifications
        oh_map: One-hot mappings
        ord_map: Ordinal mappings
        num_ranges: Numeric ranges
        device: Torch device
        print_probabilities: If True, output probability predictions before
            decoding categorical labels.
    """
    results: Dict[str, Any] = {}

    # Check if inference components are provided
    if not all(
        [
            extractor,
            meta_clin,
            meta_treat,
            gx,
            decoder_treat,
            feat_spec,
            oh_map,
            ord_map,
            num_ranges,
            device,
        ]
    ):
        print(
            "Warning: Not all inference components provided. Falling back to observational analysis only."
        )
        use_inference = False
    else:
        use_inference = True
        print("Using model inference for counterfactual outcome generation.")

    # Compute survival columns if they don't exist
    time_col = "time_difference_years"
    event_col = "event"

    if time_col not in stage_iii_df.columns or event_col not in stage_iii_df.columns:
        print("Computing survival columns...")
        stage_iii_df = compute_survival_columns(stage_iii_df)

    # Step 1: Patient cohort preparation
    print("\n=== Stage III Patient Cohort Preparation ===")
    print(f"Total Stage III patients: {len(stage_iii_df)}")

    # Get unique patient IDs for inference
    patient_ids = stage_iii_df["_id"].unique()
    print(f"Unique patients for inference: {len(patient_ids)}")

    # Step 2: Generate counterfactual outcomes using model inference
    if use_inference:
        print("\n=== Generating Counterfactual Outcomes ===")

        # Define treatment scenarios for comparison
        treatment_scenarios = {
            "baseline": [0, 0, 0],  # No treatment
            "S": [1, 0, 0],  # Surgery only
            "S_CT": [1, 1, 0],  # Surgery + Chemotherapy
            "S_RT": [1, 0, 1],  # Surgery + Radiotherapy
            "S_RT_CT": [1, 1, 1],  # Surgery + RT + CT
        }

        generated_outcomes = {}

        # Track total successes and failures for debugging
        total_successes = 0
        total_failures = 0

        # Prepare batch dataset for all patients at once
        print("Creating batch dataset for all patients...")
        try:
            # Create a new extractor instance if needed
            if extractor is None:
                extractor = MongoExtractor(
                    connection_string=os.getenv("MONGO_URI"),
                    database_name=os.getenv("MONGO_DB"),
                    collection_name=os.getenv("MONGO_COLLECTION"),
                    config_file="Models/configs/temporal_inference.yaml",
                )

            # Get data for all patients at once using the stage_iii_df
            df_all_patients = stage_iii_df.copy()

            # Create dataset and loader for all patients (limit batch size to 48 for better inference)
            dataset = TabularDatasetPID(
                df_all_patients, meta_clin, meta_treat, seq_len=5
            )
            loader = DataLoader(
                dataset, batch_size=min(48, len(patient_ids)), shuffle=False
            )

            # Collect embeddings from all batches
            all_embeddings = []
            all_patient_batches = []

            for batch_data in loader:
                _, clin_emb_batch, _ = batch_data
                clin_emb_batch = clin_emb_batch.to(device)
                all_embeddings.append(clin_emb_batch)
                all_patient_batches.append(clin_emb_batch.shape[0])

            # Concatenate all embeddings
            clin_emb_all = torch.cat(all_embeddings, dim=0)
            print(
                f"Successfully created embeddings for {clin_emb_all.shape[0]} patients"
            )

            for scenario_name, treatment_vector in treatment_scenarios.items():
                print(f"Generating outcomes for {scenario_name} treatment...")

                try:
                    # Generate counterfactual outcomes in batches of 48 for better inference stability
                    scenario_outcomes = []
                    scenario_successes = 0
                    scenario_failures = 0

                    inference_batch_size = 48
                    num_batches = (
                        clin_emb_all.shape[0] + inference_batch_size - 1
                    ) // inference_batch_size

                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * inference_batch_size
                        end_idx = min(
                            start_idx + inference_batch_size, clin_emb_all.shape[0]
                        )

                        # Get current batch embeddings
                        clin_emb_batch = clin_emb_all[start_idx:end_idx]
                        current_batch_size = clin_emb_batch.shape[0]
                        seq_len = clin_emb_batch.shape[1]

                        # Create treatment tensor for current batch
                        treat_tensor = torch.zeros(
                            (current_batch_size, seq_len, 3),
                            dtype=torch.float32,
                            device=device,
                        )
                        treat_tensor[:, 0, :] = (
                            torch.tensor(
                                treatment_vector, dtype=torch.float32, device=device
                            )
                            .unsqueeze(0)
                            .expand(current_batch_size, -1)
                        )

                        # Generate counterfactual outcomes for current batch
                        with torch.no_grad():
                            fake_outcomes_batch = gx(clin_emb_batch, treat_tensor)

                        # Process each patient in the current batch
                        for i in range(current_batch_size):
                            patient_idx = start_idx + i
                            pid = patient_ids[patient_idx]

                            try:
                                # Optionally display probabilities for each variable
                                if print_probabilities:
                                    prob_df = decode_embedding(
                                        decoder_treat(fake_outcomes_batch[i, 3]),
                                        feat_spec,
                                        oh_map,
                                        ord_map,
                                        num_ranges,
                                        one_hot_argmax=False,
                                    )
                                    print(
                                        f"\nProbabilities for patient {pid} "
                                        f"scenario {scenario_name}:"
                                    )
                                    print(prob_df.to_string(index=False))

                                # Decode the final outcome for patient i
                                decoded_outcome = decode_embedding(
                                    decoder_treat(fake_outcomes_batch[i, 3]),
                                    feat_spec,
                                    oh_map,
                                    ord_map,
                                    num_ranges,
                                )

                                # Check if decoded outcome is valid
                                if decoded_outcome.empty:
                                    scenario_failures += 1
                                    continue

                                # Extract relevant outcome metrics and add patient ID
                                outcome_dict = decoded_outcome.to_dict("records")[0]
                                outcome_dict["_id"] = pid
                                outcome_dict["treatment_scenario"] = scenario_name
                                outcome_dict["surgery"] = treatment_vector[0]
                                outcome_dict["chemotherapy"] = treatment_vector[1]
                                outcome_dict["radiotherapy"] = treatment_vector[2]

                                # Store original sarculator scores for comparison
                                original_patient = stage_iii_df[
                                    stage_iii_df["_id"] == pid
                                ].iloc[0]
                                if "general.sarculator_five" in original_patient:
                                    outcome_dict["original_sarculator_five"] = (
                                        original_patient.get("general.sarculator_five")
                                    )
                                if "general.sarculator_ten" in original_patient:
                                    outcome_dict["original_sarculator_ten"] = (
                                        original_patient.get("general.sarculator_ten")
                                    )

                                scenario_outcomes.append(outcome_dict)
                                scenario_successes += 1

                            except Exception as e:
                                print(
                                    f"Debug: Failed to decode outcome for patient {pid} in scenario {scenario_name}: {e}"
                                )
                                scenario_failures += 1
                                continue

                    # Print sample of decoded outcomes for first scenario for verification
                    if scenario_name == "S" and scenario_outcomes:
                        print(
                            f"\nSample decoded outcome for first patient in scenario {scenario_name}:"
                        )
                        sample_outcome = scenario_outcomes[0]
                        # Create a clean sample without internal keys
                        sample_display = {
                            k: v
                            for k, v in sample_outcome.items()
                            if not k.startswith("_")
                            and k
                            not in [
                                "treatment_scenario",
                                "surgery",
                                "chemotherapy",
                                "radiotherapy",
                            ]
                        }
                        vertical = list(sample_display.items())
                        print(
                            tabulate(
                                vertical,
                                headers=["variable", "value"],
                                tablefmt="grid",
                                floatfmt=".4f",
                            )
                        )

                    generated_outcomes[scenario_name] = pd.DataFrame(scenario_outcomes)
                    print(
                        f"Generated {len(scenario_outcomes)} outcomes for {scenario_name} (successes: {scenario_successes}, failures: {scenario_failures})"
                    )
                    total_successes += scenario_successes
                    total_failures += scenario_failures

                except Exception as e:
                    print(
                        f"Error generating outcomes for scenario {scenario_name}: {e}"
                    )
                    generated_outcomes[scenario_name] = pd.DataFrame()
                    total_failures += len(patient_ids)

        except Exception as e:
            print(f"Error creating batch dataset: {e}")
            print("Optimizing for batch patient inference...")

            # Optimize batch processing for all patients across all scenarios
            for scenario_name, treatment_vector in treatment_scenarios.items():
                print(
                    f"Processing all patients in batch for scenario {scenario_name}..."
                )
                scenario_outcomes = []
                scenario_successes = 0
                scenario_failures = 0

                try:
                    # Use the entire stage_iii_df for batch processing
                    df_batch = stage_iii_df.copy()

                    # Create batch dataset and loader for all patients (limit batch size to 48 for better inference)
                    dataset_batch = TabularDatasetPID(
                        df_batch, meta_clin, meta_treat, seq_len=5
                    )
                    loader_batch = DataLoader(
                        dataset_batch,
                        batch_size=min(48, len(patient_ids)),
                        shuffle=False,
                    )

                    # Collect embeddings from all batches
                    all_embeddings_batch = []
                    for batch_data in loader_batch:
                        _, clin_emb_chunk, _ = batch_data
                        clin_emb_chunk = clin_emb_chunk.to(device)
                        all_embeddings_batch.append(clin_emb_chunk)

                    # Concatenate all embeddings
                    clin_emb_batch = torch.cat(all_embeddings_batch, dim=0)

                    batch_size = clin_emb_batch.shape[0]
                    seq_len = clin_emb_batch.shape[1]

                    print(
                        f"Batch processing {batch_size} patients with sequence length {seq_len}"
                    )

                    # Generate counterfactual outcomes in batches of 48 for better inference stability
                    inference_batch_size = 48
                    num_inference_batches = (
                        batch_size + inference_batch_size - 1
                    ) // inference_batch_size

                    all_fake_outcomes = []
                    for inf_batch_idx in range(num_inference_batches):
                        inf_start_idx = inf_batch_idx * inference_batch_size
                        inf_end_idx = min(
                            inf_start_idx + inference_batch_size, batch_size
                        )

                        # Get current inference batch
                        clin_emb_inf_batch = clin_emb_batch[inf_start_idx:inf_end_idx]
                        current_inf_batch_size = clin_emb_inf_batch.shape[0]

                        # Create treatment tensor for current inference batch
                        treat_tensor_inf_batch = torch.zeros(
                            (current_inf_batch_size, seq_len, 3),
                            dtype=torch.float32,
                            device=device,
                        )
                        treat_tensor_inf_batch[:, :, :] = (
                            torch.tensor(
                                treatment_vector, dtype=torch.float32, device=device
                            )
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .expand(current_inf_batch_size, seq_len, -1)
                        )

                        # Generate counterfactual outcomes for current inference batch
                        with torch.no_grad():
                            fake_outcomes_inf_batch = gx(
                                clin_emb_inf_batch, treat_tensor_inf_batch
                            )

                        all_fake_outcomes.append(fake_outcomes_inf_batch)

                    # Concatenate all inference results
                    fake_outcomes_batch = torch.cat(all_fake_outcomes, dim=0)

                    print(
                        f"Generated counterfactual outcomes tensor shape: {fake_outcomes_batch.shape}"
                    )

                    # Decode outcomes for all patients in parallel chunks
                    chunk_size = 50  # Process 50 patients at a time to manage memory
                    num_chunks = (len(patient_ids) + chunk_size - 1) // chunk_size

                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, len(patient_ids))
                        chunk_patient_ids = patient_ids[start_idx:end_idx]

                        print(
                            f"Processing chunk {chunk_idx + 1}/{num_chunks}: patients {start_idx} to {end_idx-1}"
                        )

                        for i, pid in enumerate(chunk_patient_ids):
                            patient_idx = start_idx + i
                            try:
                                if print_probabilities:
                                    prob_df = decode_embedding(
                                        decoder_treat(
                                            fake_outcomes_batch[patient_idx, -1]
                                        ),
                                        feat_spec,
                                        oh_map,
                                        ord_map,
                                        num_ranges,
                                        one_hot_argmax=False,
                                    )
                                    print(
                                        f"\nProbabilities for patient {pid} scenario {scenario_name}:"
                                    )
                                    print(prob_df.to_string(index=False))

                                # Decode the final outcome for patient i
                                decoded_outcome = decode_embedding(
                                    decoder_treat(fake_outcomes_batch[patient_idx, -1]),
                                    feat_spec,
                                    oh_map,
                                    ord_map,
                                    num_ranges,
                                )

                                # Check if decoded outcome is valid
                                if decoded_outcome.empty:
                                    scenario_failures += 1
                                    continue

                                # Extract relevant outcome metrics and add patient ID
                                outcome_dict = decoded_outcome.to_dict("records")[0]
                                outcome_dict["_id"] = pid
                                outcome_dict["treatment_scenario"] = scenario_name
                                outcome_dict["surgery"] = treatment_vector[0]
                                outcome_dict["chemotherapy"] = treatment_vector[1]
                                outcome_dict["radiotherapy"] = treatment_vector[2]

                                # Store original sarculator scores for comparison
                                original_patient = stage_iii_df[
                                    stage_iii_df["_id"] == pid
                                ].iloc[0]
                                if "general.sarculator_five" in original_patient:
                                    outcome_dict["original_sarculator_five"] = (
                                        original_patient.get("general.sarculator_five")
                                    )
                                if "general.sarculator_ten" in original_patient:
                                    outcome_dict["original_sarculator_ten"] = (
                                        original_patient.get("general.sarculator_ten")
                                    )

                                scenario_outcomes.append(outcome_dict)
                                scenario_successes += 1

                            except Exception as e:
                                print(
                                    f"Failed to decode outcome for patient {pid}: {e}"
                                )
                                scenario_failures += 1
                                continue

                    print(
                        f"Completed batch processing for {scenario_name}: {scenario_successes} successes, {scenario_failures} failures"
                    )

                except Exception as e:
                    print(f"Batch processing failed for {scenario_name}: {e}")
                    print("Falling back to individual patient processing...")

                    # Fallback to individual processing if batch completely fails
                    for pid in patient_ids:
                        try:
                            # Get patient data
                            if extractor is None:
                                extractor = MongoExtractor(
                                    connection_string=os.getenv("MONGO_URI"),
                                    database_name=os.getenv("MONGO_DB"),
                                    collection_name=os.getenv("MONGO_COLLECTION"),
                                    config_file="Models/configs/temporal_inference.yaml",
                                )
                            df_patient = extractor.get_patient_by_id(pid)
                            if df_patient.empty:
                                scenario_failures += 1
                                continue

                            # Create dataset and loader
                            dataset = TabularDatasetPID(
                                df_patient, meta_clin, meta_treat, seq_len=5
                            )
                            loader = DataLoader(dataset, batch_size=1, shuffle=False)
                            _, clin_emb, _ = next(iter(loader))
                            clin_emb = clin_emb.to(device)

                            # Create treatment tensor for this scenario
                            treat_tensor = torch.zeros(
                                (1, clin_emb.size(1), 3),
                                dtype=torch.float32,
                                device=device,
                            )
                            treat_tensor[0, :, :] = (
                                torch.tensor(
                                    treatment_vector, dtype=torch.float32, device=device
                                )
                                .unsqueeze(0)
                                .expand(clin_emb.size(1), -1)
                            )

                            # Generate counterfactual outcome
                            with torch.no_grad():
                                fake_outcomes = gx(clin_emb, treat_tensor)

                            # Optionally show probabilities for each variable
                            if print_probabilities:
                                prob_df = decode_embedding(
                                    decoder_treat(fake_outcomes[0, -1]),
                                    feat_spec,
                                    oh_map,
                                    ord_map,
                                    num_ranges,
                                    one_hot_argmax=False,
                                )
                                print(
                                    f"\nProbabilities for patient {pid} scenario {scenario_name}:"
                                )
                                print(prob_df.to_string(index=False))

                            # Decode the final outcome
                            decoded_outcome = decode_embedding(
                                decoder_treat(fake_outcomes[0, -1]),
                                feat_spec,
                                oh_map,
                                ord_map,
                                num_ranges,
                            )

                            # Check if decoded outcome is valid
                            if decoded_outcome.empty:
                                scenario_failures += 1
                                continue

                            # Extract relevant outcome metrics and add patient ID
                            outcome_dict = decoded_outcome.to_dict("records")[0]
                            outcome_dict["_id"] = pid
                            outcome_dict["treatment_scenario"] = scenario_name
                            outcome_dict["surgery"] = treatment_vector[0]
                            outcome_dict["chemotherapy"] = treatment_vector[1]
                            outcome_dict["radiotherapy"] = treatment_vector[2]

                            # Store original sarculator scores for comparison
                            original_patient = stage_iii_df[
                                stage_iii_df["_id"] == pid
                            ].iloc[0]
                            if "general.sarculator_five" in original_patient:
                                outcome_dict["original_sarculator_five"] = (
                                    original_patient.get("general.sarculator_five")
                                )
                            if "general.sarculator_ten" in original_patient:
                                outcome_dict["original_sarculator_ten"] = (
                                    original_patient.get("general.sarculator_ten")
                                )

                            scenario_outcomes.append(outcome_dict)
                            scenario_successes += 1

                        except Exception as e:
                            print(
                                f"Debug: Failed to generate outcome for patient {pid} in scenario {scenario_name}: {e}"
                            )
                            scenario_failures += 1
                            continue

                generated_outcomes[scenario_name] = pd.DataFrame(scenario_outcomes)
                print(
                    f"Generated {len(scenario_outcomes)} outcomes for {scenario_name} (successes: {scenario_successes}, failures: {scenario_failures})"
                )
                total_successes += scenario_successes
                total_failures += scenario_failures

        # Check if we have sufficient generated data
        total_generated = sum(len(df) for df in generated_outcomes.values())
        print(f"Total inference attempts: {total_successes + total_failures}")
        print(
            f"Success rate: {total_successes/(total_successes + total_failures)*100:.1f}%"
            if (total_successes + total_failures) > 0
            else "Success rate: 0%"
        )

        if total_generated < len(patient_ids) * 0.1:  # Less than 10% success rate
            print(
                f"Warning: Low inference success rate ({total_generated}/{len(patient_ids)*4} outcomes generated)"
            )
            print(
                "Falling back to observational analysis with simulated treatment assignments..."
            )

            # Create synthetic treatment assignments for each patient across all scenarios
            synthetic_outcomes = []
            for scenario_name, treatment_vector in treatment_scenarios.items():
                for pid in patient_ids:
                    # Get original patient data from stage_iii_df
                    patient_data = stage_iii_df[stage_iii_df["_id"] == pid].copy()
                    if not patient_data.empty:
                        # Assign the treatment vector to this patient
                        patient_data = patient_data.iloc[
                            0:1
                        ].copy()  # Take first record only
                        patient_data["treatment_scenario"] = scenario_name
                        patient_data["episodes.surgery"] = treatment_vector[0]
                        patient_data["episodes.chemotherapy"] = treatment_vector[1]
                        patient_data["episodes.radiotherapy"] = treatment_vector[2]
                        synthetic_outcomes.append(patient_data)

            if synthetic_outcomes:
                analysis_df = pd.concat(synthetic_outcomes, ignore_index=True)
                print(
                    f"Created synthetic dataset with {len(analysis_df)} patient-treatment combinations"
                )
                results["used_inference"] = False
                results["fallback_reason"] = "Low inference success rate"
            else:
                print("Fallback to original observational data")
                analysis_df = stage_iii_df
                results["used_inference"] = False
                results["fallback_reason"] = "Failed to create synthetic data"

        else:
            # Convert to format compatible with existing analysis
            stage_iii_df_generated = pd.concat(
                [
                    df.assign(
                        **{
                            "episodes.surgery": df["surgery"],
                            "episodes.chemotherapy": df["chemotherapy"],
                            "episodes.radiotherapy": df["radiotherapy"],
                        }
                    )
                    for df in generated_outcomes.values()
                    if not df.empty
                ],
                ignore_index=True,
            )

            print(f"Total generated outcomes: {len(stage_iii_df_generated)}")

            # Compare sarculator scores across treatment scenarios to assess treatment effectiveness
            print("\n=== Sarculator Score Analysis by Treatment Scenario ===")
            sarculator_comparison = {}

            # Analyze sarculator scores by treatment scenario
            if (
                "general.sarculator_five" in stage_iii_df_generated.columns
                and "treatment_scenario" in stage_iii_df_generated.columns
            ):
                scenario_analysis_five = {}

                for scenario in treatment_scenarios.keys():
                    scenario_data = stage_iii_df_generated[
                        (stage_iii_df_generated["treatment_scenario"] == scenario)
                        & stage_iii_df_generated["general.sarculator_five"].notna()
                    ]

                    if len(scenario_data) > 0:
                        scenario_scores = scenario_data[
                            "general.sarculator_five"
                        ].values
                        scenario_analysis_five[scenario] = {
                            "mean": np.mean(scenario_scores),
                            "std": np.std(scenario_scores),
                            "median": np.median(scenario_scores),
                            "n": len(scenario_scores),
                        }

                print(f"  Sarculator 5-year survival by treatment scenario:")
                for scenario, stats in scenario_analysis_five.items():
                    print(
                        f"    {scenario}: mean={stats['mean']:.3f}±{stats['std']:.3f}, median={stats['median']:.3f} (n={stats['n']})"
                    )

                sarculator_comparison["five_year_by_scenario"] = scenario_analysis_five

                # Find best and worst performing scenarios
                if scenario_analysis_five:
                    best_scenario_five = max(
                        scenario_analysis_five.keys(),
                        key=lambda x: scenario_analysis_five[x]["mean"],
                    )
                    worst_scenario_five = min(
                        scenario_analysis_five.keys(),
                        key=lambda x: scenario_analysis_five[x]["mean"],
                    )
                    improvement_five = (
                        scenario_analysis_five[best_scenario_five]["mean"]
                        - scenario_analysis_five[worst_scenario_five]["mean"]
                    )

                    sarculator_comparison["best_scenario_five"] = best_scenario_five
                    sarculator_comparison["worst_scenario_five"] = worst_scenario_five
                    sarculator_comparison["improvement_five"] = improvement_five

                    print(
                        f"    Best performing: {best_scenario_five} (mean={scenario_analysis_five[best_scenario_five]['mean']:.3f})"
                    )
                    print(
                        f"    Worst performing: {worst_scenario_five} (mean={scenario_analysis_five[worst_scenario_five]['mean']:.3f})"
                    )
                    print(
                        f"    Absolute improvement: {improvement_five:.3f} ({improvement_five*100:.1f}%)"
                    )

                    # Statistical significance testing between best and worst
                    if len(scenario_analysis_five) >= 2:
                        best_data = stage_iii_df_generated[
                            (
                                stage_iii_df_generated["treatment_scenario"]
                                == best_scenario_five
                            )
                            & stage_iii_df_generated["general.sarculator_five"].notna()
                        ]["general.sarculator_five"].values

                        worst_data = stage_iii_df_generated[
                            (
                                stage_iii_df_generated["treatment_scenario"]
                                == worst_scenario_five
                            )
                            & stage_iii_df_generated["general.sarculator_five"].notna()
                        ]["general.sarculator_five"].values

                        if len(best_data) > 5 and len(worst_data) > 5:
                            try:
                                t_stat, p_value = ttest_ind(best_data, worst_data)
                                sarculator_comparison["five_year_ttest_p"] = p_value
                                print(
                                    f"    T-test p-value (best vs worst): {p_value:.3f} ({'Significant' if p_value < 0.05 else 'Not significant'})"
                                )
                            except ImportError:
                                print("    scipy not available for statistical testing")

            if (
                "general.sarculator_ten" in stage_iii_df_generated.columns
                and "treatment_scenario" in stage_iii_df_generated.columns
            ):
                scenario_analysis_ten = {}

                for scenario in treatment_scenarios.keys():
                    scenario_data = stage_iii_df_generated[
                        (stage_iii_df_generated["treatment_scenario"] == scenario)
                        & stage_iii_df_generated["general.sarculator_ten"].notna()
                    ]

                    if len(scenario_data) > 0:
                        scenario_scores = scenario_data["general.sarculator_ten"].values
                        scenario_analysis_ten[scenario] = {
                            "mean": np.mean(scenario_scores),
                            "std": np.std(scenario_scores),
                            "median": np.median(scenario_scores),
                            "n": len(scenario_scores),
                        }

                print(f"  Sarculator 10-year survival by treatment scenario:")
                for scenario, stats in scenario_analysis_ten.items():
                    print(
                        f"    {scenario}: mean={stats['mean']:.3f}±{stats['std']:.3f}, median={stats['median']:.3f} (n={stats['n']})"
                    )

                sarculator_comparison["ten_year_by_scenario"] = scenario_analysis_ten

                # Find best and worst performing scenarios
                if scenario_analysis_ten:
                    best_scenario_ten = max(
                        scenario_analysis_ten.keys(),
                        key=lambda x: scenario_analysis_ten[x]["mean"],
                    )
                    worst_scenario_ten = min(
                        scenario_analysis_ten.keys(),
                        key=lambda x: scenario_analysis_ten[x]["mean"],
                    )
                    improvement_ten = (
                        scenario_analysis_ten[best_scenario_ten]["mean"]
                        - scenario_analysis_ten[worst_scenario_ten]["mean"]
                    )

                    sarculator_comparison["best_scenario_ten"] = best_scenario_ten
                    sarculator_comparison["worst_scenario_ten"] = worst_scenario_ten
                    sarculator_comparison["improvement_ten"] = improvement_ten

                    print(
                        f"    Best performing: {best_scenario_ten} (mean={scenario_analysis_ten[best_scenario_ten]['mean']:.3f})"
                    )
                    print(
                        f"    Worst performing: {worst_scenario_ten} (mean={scenario_analysis_ten[worst_scenario_ten]['mean']:.3f})"
                    )
                    print(
                        f"    Absolute improvement: {improvement_ten:.3f} ({improvement_ten*100:.1f}%)"
                    )

            # Store comparison results
            results["sarculator_comparison"] = sarculator_comparison

            # Create patient-level comparison table
            print("\n=== Patient-Level Sarculator Scores by Treatment Scenario ===")

            # Create pivot tables for sarculator scores by patient and treatment scenario
            if "general.sarculator_five" in stage_iii_df_generated.columns:
                # Pivot table for 5-year sarculator scores
                pivot_five = stage_iii_df_generated.pivot_table(
                    index="_id",
                    columns="treatment_scenario",
                    values="general.sarculator_five",
                    aggfunc="first",  # Use first value in case of duplicates
                )

                # Fill missing combinations with NaN and format for display
                if not pivot_five.empty:
                    print("\n5-Year Sarculator Scores by Patient and Treatment:")
                    # Format the table for better readability
                    pivot_five_formatted = pivot_five.round(4)
                    print(
                        tabulate(
                            pivot_five_formatted,
                            headers=pivot_five_formatted.columns,
                            tablefmt="grid",
                            floatfmt=".4f",
                            showindex=True,
                        )
                    )

                    # Calculate differences between treatment scenarios relative to baseline (S)
                    baseline_col = "S"  # Surgery only as baseline
                    if baseline_col in pivot_five_formatted.columns:
                        # Define treatment columns to compare (exclude baseline and 'baseline' scenario)
                        treatment_cols = [
                            col
                            for col in pivot_five_formatted.columns
                            if col != baseline_col
                            and col in ["S_CT", "S_RT", "S_RT_CT"]
                        ]

                        # Compare each treatment to baseline
                        for col in treatment_cols:
                            # Absolute difference from baseline
                            diff_col = f"{col}_diff"
                            pivot_five_formatted[diff_col] = (
                                pivot_five_formatted[col]
                                - pivot_five_formatted[baseline_col]
                            )

                            # Percentage difference from baseline
                            pct_col = f"{col}_pct"
                            pivot_five_formatted[pct_col] = (
                                (
                                    pivot_five_formatted[col]
                                    - pivot_five_formatted[baseline_col]
                                )
                                / pivot_five_formatted[baseline_col]
                                * 100
                            ).round(2)

                    # Show differences table if calculated
                    diff_cols = [
                        col
                        for col in pivot_five_formatted.columns
                        if "_diff" in col or "_pct" in col
                    ]
                    if diff_cols:
                        # Sort columns for better readability: group by treatment, then diff/pct
                        diff_cols_sorted = []
                        for treatment in ["S_CT", "S_RT", "S_RT_CT"]:
                            diff_col = f"{treatment}_diff"
                            pct_col = f"{treatment}_pct"
                            if diff_col in diff_cols:
                                diff_cols_sorted.extend([diff_col, pct_col])

                        print(f"\n5-Year Sarculator Score Differences vs Baseline (S):")
                        print("(diff = absolute difference, pct = percentage change)")
                        print(
                            tabulate(
                                pivot_five_formatted[diff_cols_sorted],
                                headers=diff_cols_sorted,
                                tablefmt="grid",
                                floatfmt=".4f",
                                showindex=True,
                            )
                        )

                        # Save 5-year pivot results
                        results["pivot_five_year"] = pivot_five_formatted.to_dict()
                        results["pivot_five_year_diff_cols"] = diff_cols_sorted

            if "general.sarculator_ten" in stage_iii_df_generated.columns:
                # Pivot table for 10-year sarculator scores
                pivot_ten = stage_iii_df_generated.pivot_table(
                    index="_id",
                    columns="treatment_scenario",
                    values="general.sarculator_ten",
                    aggfunc="first",  # Use first value in case of duplicates
                )

                # Fill missing combinations with NaN and format for display
                if not pivot_ten.empty:
                    print("\n10-Year Sarculator Scores by Patient and Treatment:")
                    # Format the table for better readability
                    pivot_ten_formatted = pivot_ten.round(4)
                    print(
                        tabulate(
                            pivot_ten_formatted,
                            headers=pivot_ten_formatted.columns,
                            tablefmt="grid",
                            floatfmt=".4f",
                            showindex=True,
                        )
                    )

                    # Calculate differences between treatment scenarios relative to baseline (S)
                    baseline_col = "S"  # Surgery only as baseline
                    if baseline_col in pivot_ten_formatted.columns:
                        # Define treatment columns to compare (exclude baseline and 'baseline' scenario)
                        treatment_cols = [
                            col
                            for col in pivot_ten_formatted.columns
                            if col != baseline_col
                            and col in ["S_CT", "S_RT", "S_RT_CT"]
                        ]

                        # Compare each treatment to baseline
                        for col in treatment_cols:
                            # Absolute difference from baseline
                            diff_col = f"{col}_diff"
                            pivot_ten_formatted[diff_col] = (
                                pivot_ten_formatted[col]
                                - pivot_ten_formatted[baseline_col]
                            )

                            # Percentage difference from baseline
                            pct_col = f"{col}_pct"
                            pivot_ten_formatted[pct_col] = (
                                (
                                    pivot_ten_formatted[col]
                                    - pivot_ten_formatted[baseline_col]
                                )
                                / pivot_ten_formatted[baseline_col]
                                * 100
                            ).round(2)

                    # Show differences table if calculated
                    diff_cols = [
                        col
                        for col in pivot_ten_formatted.columns
                        if "_diff" in col or "_pct" in col
                    ]
                    if diff_cols:
                        # Sort columns for better readability: group by treatment, then diff/pct
                        diff_cols_sorted = []
                        for treatment in ["S_CT", "S_RT", "S_RT_CT"]:
                            diff_col = f"{treatment}_diff"
                            pct_col = f"{treatment}_pct"
                            if diff_col in diff_cols:
                                diff_cols_sorted.extend([diff_col, pct_col])

                        print(
                            f"\n10-Year Sarculator Score Differences vs Baseline (S):"
                        )
                        print("(diff = absolute difference, pct = percentage change)")
                        print(
                            tabulate(
                                pivot_ten_formatted[diff_cols_sorted],
                                headers=diff_cols_sorted,
                                tablefmt="grid",
                                floatfmt=".4f",
                                showindex=True,
                            )
                        )

                        # Save 10-year pivot results
                        results["pivot_ten_year"] = pivot_ten_formatted.to_dict()
                        results["pivot_ten_year_diff_cols"] = diff_cols_sorted

            # Create sarculator comparison plot by treatment scenario
            plot_sarculator_by_scenario(
                stage_iii_df_generated, "sarculator_comparison_stage_iii.png"
            )

            # Save comprehensive CSV file with original patient data and generated outcomes
            print("\n=== Saving Comprehensive Dataset ===")
            try:
                # Create a comprehensive dataset with original and generated data
                comprehensive_data = []

                # Get unique patient IDs
                unique_patients = stage_iii_df_generated["_id"].unique()

                for pid in unique_patients:
                    # Get original patient data
                    original_data = (
                        stage_iii_df[stage_iii_df["_id"] == pid].iloc[0].to_dict()
                    )

                    # Get all generated scenarios for this patient
                    patient_scenarios = stage_iii_df_generated[
                        stage_iii_df_generated["_id"] == pid
                    ]

                    for _, scenario_row in patient_scenarios.iterrows():
                        # Create combined record
                        combined_record = {}

                        # Add patient ID and original baseline characteristics
                        combined_record["patient_id"] = pid
                        combined_record["original_age"] = original_data.get(
                            "general.age", None
                        )
                        combined_record["original_sarculator_five"] = original_data.get(
                            "general.sarculator_five", None
                        )
                        combined_record["original_sarculator_ten"] = original_data.get(
                            "general.sarculator_ten", None
                        )
                        combined_record["original_tumor_size"] = original_data.get(
                            "episodes.treatments.fields.tumor_max_size", None
                        )
                        combined_record["original_fnclcc_grade"] = original_data.get(
                            "tumor_characteristics.grading_fnclcc", None
                        )
                        combined_record["original_histology"] = original_data.get(
                            "tumor_characteristics.histological_diagnosis", None
                        )
                        combined_record["original_site"] = original_data.get(
                            "tumor_characteristics.lesion_site", None
                        )
                        combined_record["original_cci"] = original_data.get(
                            "episodes.diagnosis.fields.cci", None
                        )

                        # Add treatment scenario information
                        combined_record["treatment_scenario"] = scenario_row[
                            "treatment_scenario"
                        ]
                        combined_record["surgery"] = scenario_row.get(
                            "surgery", scenario_row.get("episodes.surgery", None)
                        )
                        combined_record["chemotherapy"] = scenario_row.get(
                            "chemotherapy",
                            scenario_row.get("episodes.chemotherapy", None),
                        )
                        combined_record["radiotherapy"] = scenario_row.get(
                            "radiotherapy",
                            scenario_row.get("episodes.radiotherapy", None),
                        )

                        # Add generated outcomes
                        combined_record["generated_sarculator_five"] = scenario_row.get(
                            "general.sarculator_five", None
                        )
                        combined_record["generated_sarculator_ten"] = scenario_row.get(
                            "general.sarculator_ten", None
                        )

                        # Calculate changes from original
                        if (
                            combined_record["original_sarculator_five"] is not None
                            and combined_record["generated_sarulator_five"] is not None
                        ):
                            combined_record["sarculator_five_change"] = (
                                combined_record["generated_sarculator_five"]
                                - combined_record["original_sarculator_five"]
                            )
                            combined_record["sarculator_five_pct_change"] = (
                                (
                                    combined_record["sarulator_five_change"]
                                    / combined_record["original_sarculator_five"]
                                    * 100
                                )
                                if combined_record["original_sarculator_five"] != 0
                                else 0
                            )

                        if (
                            combined_record["original_sarculator_ten"] is not None
                            and combined_record["generated_sarulator_ten"] is not None
                        ):
                            combined_record["sarculator_ten_change"] = (
                                combined_record["generated_sarculator_ten"]
                                - combined_record["original_sarculator_ten"]
                            )
                            combined_record["sarculator_ten_pct_change"] = (
                                (
                                    combined_record["sarulator_ten_change"]
                                    / combined_record["original_sarculator_ten"]
                                    * 100
                                )
                                if combined_record["original_sarulator_ten"] != 0
                                else 0
                            )

                        # Add any other relevant generated fields
                        for col in scenario_row.index:
                            if col not in [
                                "_id",
                                "treatment_scenario",
                                "surgery",
                                "chemotherapy",
                                "radiotherapy",
                                "general.sarculator_five",
                                "general.sarculator_ten",
                            ]:
                                if (
                                    col.startswith("episodes.")
                                    or col.startswith("general.")
                                    or col.startswith("tumor_")
                                ):
                                    combined_record[f"generated_{col}"] = scenario_row[
                                        col
                                    ]

                        comprehensive_data.append(combined_record)

                # Convert to DataFrame and save
                comprehensive_df = pd.DataFrame(comprehensive_data)
                csv_filename = "stage_iii_comprehensive_analysis.csv"
                comprehensive_df.to_csv(csv_filename, index=False)

                print(f"Saved comprehensive dataset to: {csv_filename}")
                print(
                    f"Dataset contains {len(comprehensive_df)} records for {len(unique_patients)} patients across {len(treatment_scenarios)} treatment scenarios"
                )
                print(f"Columns: {list(comprehensive_df.columns)}")

                # Save summary statistics
                summary_stats = {
                    "total_patients": len(unique_patients),
                    "total_records": len(comprehensive_df),
                    "treatment_scenarios": list(treatment_scenarios.keys()),
                    "mean_original_sarculator_five": comprehensive_df[
                        "original_sarculator_five"
                    ].mean(),
                    "mean_original_sarculator_ten": comprehensive_df[
                        "original_sarulator_ten"
                    ].mean(),
                }

                # Calculate mean generated scores by treatment scenario
                for scenario in treatment_scenarios.keys():
                    scenario_data = comprehensive_df[
                        comprehensive_df["treatment_scenario"] == scenario
                    ]
                    if len(scenario_data) > 0:
                        summary_stats[f"mean_generated_sarculator_five_{scenario}"] = (
                            scenario_data["generated_sarculator_five"].mean()
                        )
                        summary_stats[f"mean_generated_sarculator_ten_{scenario}"] = (
                            scenario_data["generated_sarculator_ten"].mean()
                        )

                results["comprehensive_dataset_saved"] = True
                results["comprehensive_dataset_filename"] = csv_filename
                results["dataset_summary"] = summary_stats

            except Exception as e:
                print(f"Error saving comprehensive dataset: {e}")
                results["comprehensive_dataset_saved"] = False
                results["dataset_error"] = str(e)

            # Use generated data for analysis
            analysis_df = stage_iii_df_generated
            results["used_inference"] = True

    else:
        # Use original observational data
        analysis_df = stage_iii_df
        results["used_inference"] = False

        # Save observational dataset for reference
        print("\n=== Saving Observational Dataset ===")
        try:
            csv_filename = "stage_iii_observational_analysis.csv"
            analysis_df.to_csv(csv_filename, index=False)

            print(f"Saved observational dataset to: {csv_filename}")
            print(f"Dataset contains {len(analysis_df)} records")

            results["observational_dataset_saved"] = True
            results["observational_dataset_filename"] = csv_filename

        except Exception as e:
            print(f"Error saving observational dataset: {e}")
            results["observational_dataset_saved"] = False
            results["dataset_error"] = str(e)

    # Step 3: Cohort segmentation (CONSORT-style)
    print("\n=== Cohort Segmentation (CONSORT-style) ===")
    arms, counts = cohort_segments(analysis_df)
    print("AJCC Stage III Cohort counts:")
    for arm, count in counts.items():
        print(f"  {arm}: {count} patients")
    results["cohort_counts"] = counts
    results["total_stage_iii"] = len(analysis_df)

    # Create CONSORT flow diagram
    create_consort_flow_diagram(
        stage_iii_df, analysis_df, arms, "consort_stage_iii_flow.png"
    )

    # Define covariates for balance assessment using actual field names from config
    covars = [
        col
        for col in [
            "episodes.treatments.fields.tumor_max_size",  # proxy for tumor size
            "tumor_characteristics.grading_fnclcc",  # FNCLCC grade
            "tumor_characteristics.histological_diagnosis",  # histology type
            "tumor_characteristics.lesion_site",  # anatomical location
            "general.sarculator_five",  # sarculator 5-year score
            "general.sarculator_ten",  # sarculator 10-year score
            "general.age",  # patient age
            "episodes.diagnosis.fields.cci",  # comorbidity index
        ]
        if col in analysis_df.columns
    ]

    print(f"\nAvailable covariates for balance assessment: {covars}")

    # Step 4: Baseline balance within each backbone pair
    print("\n=== Baseline Balance Assessment ===")

    # For generated data, we'll still assess balance but it may be perfect due to counterfactual nature
    balance_note = (
        " (from generated counterfactuals)"
        if use_inference
        else " (from observational data)"
    )

    # Pair 1: S vs S + CT (surgery backbone)
    if len(arms["S"]) > 0 and len(arms["S_CT"]) > 0:
        print(f"\nAnalyzing S vs S + CT comparison{balance_note}...")
        pair1 = pd.concat([arms["S"], arms["S_CT"]]).copy()
        pair1["treat"] = (pair1["episodes.chemotherapy"] == 1).astype(int)

        # For generated data, add survival outcomes from the generated results
        if use_inference and time_col not in pair1.columns:
            print("Computing survival outcomes from generated data...")
            pair1 = compute_survival_columns(pair1)

        smd1 = baseline_balance(pair1, covars, "treat")
        love_plot(smd1, "love_S_vs_SCT_stage_iii.png")
        results["smd_s_vs_sct"] = smd1
        results["pair1_n"] = len(pair1)
        print(f"  Total patients in S vs S+CT comparison: {len(pair1)}")

        # Check balance quality
        max_smd1 = max(smd1.values()) if smd1 else float("inf")
        results["pair1_balanced"] = max_smd1 < 0.1
        print(
            f"  Maximum SMD: {max_smd1:.3f} ({'Balanced' if max_smd1 < 0.1 else 'Imbalanced'})"
        )
    else:
        print("Insufficient data for S vs S+CT comparison")
        results["smd_s_vs_sct"] = {}
        results["pair1_n"] = 0
        results["pair1_balanced"] = False

    # Pair 2: S + RT vs S + RT + CT (trimodal backbone)
    if len(arms["S_RT"]) > 0 and len(arms["S_RT_CT"]) > 0:
        print(f"\nAnalyzing S + RT vs S + RT + CT comparison{balance_note}...")
        pair2 = pd.concat([arms["S_RT"], arms["S_RT_CT"]]).copy()
        pair2["treat"] = (pair2["episodes.chemotherapy"] == 1).astype(int)

        # For generated data, add survival outcomes from the generated results
        if use_inference and time_col not in pair2.columns:
            print("Computing survival outcomes from generated data...")
            pair2 = compute_survival_columns(pair2)

        smd2 = baseline_balance(pair2, covars, "treat")
        love_plot(smd2, "love_SRT_vs_SRTCT_stage_iii.png")
        results["smd_srt_vs_srtct"] = smd2
        results["pair2_n"] = len(pair2)
        print(f"  Total patients in S+RT vs S+RT+CT comparison: {len(pair2)}")

        # Check balance quality
        max_smd2 = max(smd2.values()) if smd2 else float("inf")
        results["pair2_balanced"] = max_smd2 < 0.1
        print(
            f"  Maximum SMD: {max_smd2:.3f} ({'Balanced' if max_smd2 < 0.1 else 'Imbalanced'})"
        )
    else:
        print("Insufficient data for S+RT vs S+RT+CT comparison")
        results["smd_srt_vs_srtct"] = {}
        results["pair2_n"] = 0
        results["pair2_balanced"] = False

    # Step 5: Average Treatment Effect (ATE) analysis
    print(f"\n=== Average Treatment Effect Analysis{balance_note} ===")

    if time_col in analysis_df.columns and event_col in analysis_df.columns:
        # ATE for Pair 1 (S vs S + CT)
        if results["pair1_n"] > 0:
            print("Computing ATE for S vs S+CT...")
            ate_metrics1 = km_and_hazard(
                pair1, time_col, event_col, "treat", "stage_iii_pair1"
            )
            results["ate_pair1"] = ate_metrics1

            # 5-year survival rates
            s_only = pair1[pair1["treat"] == 0]
            s_ct = pair1[pair1["treat"] == 1]
            if len(s_only) > 0 and len(s_ct) > 0:
                surv_5y_s = survival_rate_at(
                    5.0, s_only[time_col].values, s_only[event_col].values
                )
                surv_5y_sct = survival_rate_at(
                    5.0, s_ct[time_col].values, s_ct[event_col].values
                )
                results["survival_5yr_s_only"] = surv_5y_s
                results["survival_5yr_s_ct"] = surv_5y_sct
                results["absolute_benefit_s_ct"] = surv_5y_sct - surv_5y_s
                print(f"  5-year survival S only: {surv_5y_s:.1%}")
                print(f"  5-year survival S+CT: {surv_5y_sct:.1%}")
                print(f"  Absolute benefit from CT: {surv_5y_sct - surv_5y_s:.1%}")

        # ATE for Pair 2 (S + RT vs S + RT + CT)
        if results["pair2_n"] > 0:
            print("Computing ATE for S+RT vs S+RT+CT...")
            ate_metrics2 = km_and_hazard(
                pair2, time_col, event_col, "treat", "stage_iii_pair2"
            )
            results["ate_pair2"] = ate_metrics2

            # 5-year survival rates
            srt_only = pair2[pair2["treat"] == 0]
            srt_ct = pair2[pair2["treat"] == 1]
            if len(srt_only) > 0 and len(srt_ct) > 0:
                surv_5y_srt = survival_rate_at(
                    5.0, srt_only[time_col].values, srt_only[event_col].values
                )
                surv_5y_srtct = survival_rate_at(
                    5.0, srt_ct[time_col].values, srt_ct[event_col].values
                )
                results["survival_5yr_srt_only"] = surv_5y_srt
                results["survival_5yr_srt_ct"] = surv_5y_srtct
                results["absolute_benefit_srt_ct"] = surv_5y_srtct - surv_5y_srt
                print(f"  5-year survival S+RT only: {surv_5y_srt:.1%}")
                print(f"  5-year survival S+RT+CT: {surv_5y_srtct:.1%}")
                print(f"  Absolute benefit from CT: {surv_5y_srtct - surv_5y_srt:.1%}")
    else:
        print(
            f"Warning: Required columns for survival analysis not found: {time_col}, {event_col}"
        )
        results["ate_pair1"] = {}
        results["ate_pair2"] = {}

    # Step 6: Formal CT × RT interaction test
    print(f"\n=== Chemotherapy × Radiotherapy Interaction Analysis{balance_note} ===")
    interaction_results = chemo_rt_interaction(analysis_df, time_col, event_col)
    results.update(interaction_results)

    if "interaction_hr" in interaction_results:
        print(f"  Interaction HR: {interaction_results['interaction_hr']:.3f}")
        print(f"  Interaction p-value: {interaction_results['interaction_p']:.3f}")
        results["significant_interaction"] = interaction_results["interaction_p"] < 0.05

    # Step 7: Summary statistics
    print(f"\n=== Summary Statistics{balance_note} ===")
    results["analysis_feasible"] = (results["pair1_n"] > 10) or (
        results["pair2_n"] > 10
    )
    results["both_comparisons_feasible"] = (results["pair1_n"] > 10) and (
        results["pair2_n"] > 10
    )

    print(f"Analysis feasible: {results['analysis_feasible']}")
    print(f"Both comparisons feasible: {results['both_comparisons_feasible']}")

    # Clinical interpretation
    if results.get("absolute_benefit_s_ct", 0) > 0.1:  # >10% absolute benefit
        analysis_type = "counterfactual" if use_inference else "observational"
        print(
            f"  FINDING: Clinically meaningful benefit from CT in surgery-only backbone ({analysis_type} analysis)"
        )
    if results.get("absolute_benefit_srt_ct", 0) > 0.1:
        analysis_type = "counterfactual" if use_inference else "observational"
        print(
            f"  FINDING: Clinically meaningful benefit from CT in trimodal backbone ({analysis_type} analysis)"
        )

    # Add inference metadata
    if use_inference:
        results["counterfactual_analysis"] = True
        results["generated_scenarios"] = (
            list(treatment_scenarios.keys())
            if "treatment_scenarios" in locals()
            else []
        )
        print(
            f"\nCounterfactual analysis completed using {len(patient_ids)} patients across {len(results.get('generated_scenarios', []))} treatment scenarios."
        )
    else:
        results["counterfactual_analysis"] = False
        print(
            "\nObservational analysis completed using real-world treatment assignments."
        )

    return results


def analyze_local_recurrence(
    df: pd.DataFrame,
    extractor: MongoExtractor = None,
    meta_clin: MetadataHandler = None,
    meta_treat: MetadataHandler = None,
    gx=None,
    decoder_treat=None,
    feat_spec=None,
    oh_map=None,
    ord_map=None,
    num_ranges=None,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Analyze local recurrence rates comparing surgery alone vs surgery + radiotherapy.

    Uses the 'episodes.treatments.fields.endpoint' field where:
    - 1 = local recurrence
    - 2 = metastasis
    - 3 = death

    Args:
        df: Patient dataframe
        Other args: Model inference components (optional)

    Returns:
        Dictionary with local recurrence analysis results
    """
    results: Dict[str, Any] = {}

    print("\n=== Research Question 2: Local Recurrence Analysis ===")
    print("Comparing Surgery alone vs Surgery + Radiotherapy")

    # Check if inference components are provided
    use_inference = all(
        [
            extractor,
            meta_clin,
            meta_treat,
            gx,
            decoder_treat,
            feat_spec,
            oh_map,
            ord_map,
            num_ranges,
            device,
        ]
    )

    if use_inference:
        print("Using model inference for counterfactual outcome generation.")
    else:
        print("Using observational data analysis only.")

    # Define treatment cohorts
    surgery_only = df[
        (df["episodes.surgery"] == 1)
        & (df["episodes.radiotherapy"] == 0)
        & (df["episodes.chemotherapy"] == 0)  # Pure surgery only
    ]

    surgery_rt = df[
        (df["episodes.surgery"] == 1)
        & (df["episodes.radiotherapy"] == 1)
        & (df["episodes.chemotherapy"] == 0)  # Surgery + RT without chemo
    ]

    print(f"Surgery only patients: {len(surgery_only)}")
    print(f"Surgery + RT patients: {len(surgery_rt)}")

    if len(surgery_only) == 0 or len(surgery_rt) == 0:
        print("Insufficient patients in one or both treatment groups")
        return {"error": "Insufficient data for local recurrence analysis"}

    # Store cohort sizes
    results["surgery_only_n"] = len(surgery_only)
    results["surgery_rt_n"] = len(surgery_rt)

    # Generate counterfactual outcomes if inference is available
    if use_inference:
        print("\n=== Generating Counterfactual Local Recurrence Outcomes ===")
        analysis_df = generate_counterfactual_recurrence_outcomes(
            df,
            extractor,
            meta_clin,
            meta_treat,
            gx,
            decoder_treat,
            feat_spec,
            oh_map,
            ord_map,
            num_ranges,
            device,
        )
        results["used_inference"] = True
    else:
        analysis_df = df
        results["used_inference"] = False

    # Analyze local recurrence rates
    recurrence_results = compute_local_recurrence_rates(analysis_df)
    results.update(recurrence_results)

    # Statistical comparison
    stat_results = compare_recurrence_statistically(analysis_df)
    results.update(stat_results)

    # Create visualizations
    plot_local_recurrence_analysis(analysis_df, "local_recurrence_analysis.png")

    # Time-to-recurrence analysis if time data available
    time_results = analyze_time_to_recurrence(analysis_df)
    results.update(time_results)

    return results


def generate_counterfactual_recurrence_outcomes(
    df: pd.DataFrame,
    extractor: MongoExtractor,
    meta_clin: MetadataHandler,
    meta_treat: MetadataHandler,
    gx,
    decoder_treat,
    feat_spec,
    oh_map,
    ord_map,
    num_ranges,
    device: torch.device,
    treatment_scenarios: Optional[Dict[str, List[int]]] = None,
) -> pd.DataFrame:
    """Generate counterfactual outcomes for local recurrence analysis using optimized batch processing."""

    # Get patients who had surgery (our base population)
    surgery_patients = df[df["episodes.surgery"] == 1]["_id"].unique()

    surgery_patients = df[df["episodes.surgery"] == 1]["_id"].unique()

    if treatment_scenarios is None:
        treatment_scenarios = {
            "surgery_only": [1, 0, 0],  # Surgery, no chemo, no RT
            "surgery_rt": [1, 0, 1],  # Surgery, no chemo, with RT
        }

    print(
        f"Generating counterfactual recurrence outcomes for {len(surgery_patients)} surgery patients..."
    )

    counterfactual_data = []
    total_successes = 0
    total_failures = 0

    # Optimize by processing all patients at once for each scenario
    for scenario_name, treatment_vector in treatment_scenarios.items():
        print(
            f"Processing all patients in batch for recurrence scenario {scenario_name}..."
        )
        scenario_outcomes = []
        scenario_successes = 0
        scenario_failures = 0

        try:
            # Use the surgery patients subset for batch processing
            surgery_df = df[df["_id"].isin(surgery_patients)].copy()

            # Create batch dataset and loader for all surgery patients (limit batch size to 48 for better inference)
            dataset_batch = TabularDatasetPID(
                surgery_df, meta_clin, meta_treat, seq_len=5
            )
            loader_batch = DataLoader(
                dataset_batch, batch_size=min(48, len(surgery_patients)), shuffle=False
            )

            # Collect embeddings from all batches
            all_embeddings_surgery = []
            for batch_data in loader_batch:
                _, clin_emb_chunk, _ = batch_data
                clin_emb_chunk = clin_emb_chunk.to(device)
                all_embeddings_surgery.append(clin_emb_chunk)

            # Concatenate all embeddings
            clin_emb_batch = torch.cat(all_embeddings_surgery, dim=0)

            batch_size = clin_emb_batch.shape[0]
            seq_len = clin_emb_batch.shape[1]

            print(
                f"Batch processing {batch_size} surgery patients with sequence length {seq_len}"
            )

            # Generate counterfactual outcomes in batches of 48 for better inference stability
            inference_batch_size = 48
            num_inference_batches = (
                batch_size + inference_batch_size - 1
            ) // inference_batch_size

            all_fake_outcomes_surgery = []
            for inf_batch_idx in range(num_inference_batches):
                inf_start_idx = inf_batch_idx * inference_batch_size
                inf_end_idx = min(inf_start_idx + inference_batch_size, batch_size)

                # Get current inference batch
                clin_emb_inf_batch = clin_emb_batch[inf_start_idx:inf_end_idx]
                current_inf_batch_size = clin_emb_inf_batch.shape[0]

                # Create treatment tensor for current inference batch
                treat_tensor_inf_batch = torch.zeros(
                    (current_inf_batch_size, seq_len, 3),
                    dtype=torch.float32,
                    device=device,
                )
                treat_tensor_inf_batch[:, :, :] = (
                    torch.tensor(treatment_vector, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(current_inf_batch_size, seq_len, -1)
                )

                # Generate counterfactual outcomes for current inference batch
                with torch.no_grad():
                    fake_outcomes_inf_batch = gx(
                        clin_emb_inf_batch, treat_tensor_inf_batch
                    )

                all_fake_outcomes_surgery.append(fake_outcomes_inf_batch)

            # Concatenate all inference results
            fake_outcomes_batch = torch.cat(all_fake_outcomes_surgery, dim=0)

            print(
                f"Generated counterfactual recurrence outcomes tensor shape: {fake_outcomes_batch.shape}"
            )

            # Decode outcomes for all patients in parallel chunks
            chunk_size = 100  # Process 100 patients at a time to manage memory
            num_chunks = (len(surgery_patients) + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(surgery_patients))
                chunk_patient_ids = surgery_patients[start_idx:end_idx]

                print(
                    f"Processing recurrence chunk {chunk_idx + 1}/{num_chunks}: patients {start_idx} to {end_idx-1}"
                )

                for i, pid in enumerate(chunk_patient_ids):
                    patient_idx = start_idx + i
                    try:
                        # Decode the final outcome for patient i using endpoint-specific decoder
                        decoded_outcome = decode_embedding_endpoint(
                            decoder_treat(fake_outcomes_batch[patient_idx, -1]),
                            feat_spec,
                            oh_map,
                            ord_map,
                            num_ranges,
                        )

                        # Check if decoded outcome is valid
                        if decoded_outcome.empty:
                            scenario_failures += 1
                            continue

                        # Extract relevant outcome metrics and add patient ID
                        outcome_dict = decoded_outcome.to_dict("records")[0]
                        outcome_dict["_id"] = pid
                        outcome_dict["treatment_scenario"] = scenario_name
                        outcome_dict["episodes.surgery"] = treatment_vector[0]
                        outcome_dict["episodes.chemotherapy"] = treatment_vector[1]
                        outcome_dict["episodes.radiotherapy"] = treatment_vector[2]

                        # Add patient's clinical characteristics from the original dataframe
                        # (Assumes surgery_df is available and contains clinical columns)
                        clinical_row = surgery_df[surgery_df["_id"] == pid]
                        if not clinical_row.empty:
                            clinical_data = clinical_row.iloc[0].to_dict()
                            # Only add clinical columns (exclude treatment and outcome columns)
                            for k, v in clinical_data.items():
                                if (
                                    k not in outcome_dict
                                    and not k.startswith("episodes.")
                                    and not k.startswith("treatment_scenario")
                                ):
                                    outcome_dict[k] = v

                        scenario_outcomes.append(outcome_dict)
                        scenario_successes += 1

                    except Exception as e:
                        print(
                            f"Failed to decode recurrence outcome for patient {pid}: {e}"
                        )
                        scenario_failures += 1
                        continue

            print(
                f"Completed batch recurrence processing for {scenario_name}: {scenario_successes} successes, {scenario_failures} failures"
            )

            # Save scenario outcomes to CSV after each scenario completes
            if scenario_outcomes:
                scenario_df = pd.DataFrame(scenario_outcomes)
                csv_filename = f"counterfactual_outcomes_{scenario_name}.csv"
                scenario_df.to_csv(csv_filename, index=False)
                print(
                    f"Saved {len(scenario_outcomes)} outcomes for {scenario_name} to {csv_filename}"
                )

            # Show sample of decoded outcomes for verification
            if scenario_outcomes and scenario_name == "surgery_only":
                print(
                    f"\nSample decoded recurrence outcome for first patient in scenario {scenario_name}:"
                )
                sample_outcome = scenario_outcomes[0]

                # Show endpoint probabilities if available
                endpoint_info = {}
                if "local_recurrence_prob" in sample_outcome:
                    endpoint_info["Local Recurrence Prob"] = sample_outcome[
                        "local_recurrence_prob"
                    ]
                if "metastasis_prob" in sample_outcome:
                    endpoint_info["Metastasis Prob"] = sample_outcome["metastasis_prob"]
                if "death_prob" in sample_outcome:
                    endpoint_info["Death Prob"] = sample_outcome["death_prob"]
                if "episodes.treatments.fields.endpoint" in sample_outcome:
                    endpoint_info["Most Likely Outcome"] = sample_outcome[
                        "episodes.treatments.fields.endpoint"
                    ]
                if "endpoint_max_probability" in sample_outcome:
                    endpoint_info["Max Probability"] = sample_outcome[
                        "endpoint_max_probability"
                    ]

                if endpoint_info:
                    print("Endpoint Analysis:")
                    for key, value in endpoint_info.items():
                        print(f"  {key}: {value}")
                else:
                    print(
                        "No endpoint probability information found in decoded outcome"
                    )

        except Exception as e:
            print(f"Batch recurrence processing failed for {scenario_name}: {e}")
            print(
                "Falling back to individual patient processing for recurrence analysis..."
            )

            # Fallback to individual processing if batch completely fails
            for pid in surgery_patients:
                try:
                    extractor = MongoExtractor(
                        connection_string=os.getenv("MONGO_URI"),
                        database_name=os.getenv("MONGO_DB"),
                        collection_name=os.getenv("MONGO_COLLECTION"),
                        config_file="Models/configs/temporal_inference.yaml",
                    )
                    patient_df = extractor.get_patient_by_id(pid)
                    if patient_df.empty:
                        scenario_failures += 1
                        continue

                    # Create dataset for this patient
                    dataset = TabularDatasetPID(
                        patient_df, meta_clin, meta_treat, seq_len=5
                    )
                    loader = DataLoader(dataset, batch_size=1, shuffle=False)
                    _, clin_emb, _ = next(iter(loader))
                    clin_emb = clin_emb.to(device)

                    # Generate treatment tensor
                    treat_tensor = torch.tensor(
                        treatment_vector, dtype=torch.float32, device=device
                    )
                    treat_tensor = (
                        treat_tensor.unsqueeze(0)
                        .unsqueeze(1)
                        .repeat(1, clin_emb.size(1), 1)
                    )

                    # Generate counterfactual outcome
                    with torch.no_grad():
                        fake_outcome = gx(clin_emb, treat_tensor)

                    # Decode the final outcome using endpoint-specific decoder
                    decoded_outcome = decode_embedding_endpoint(
                        decoder_treat(fake_outcome[0, -1]),
                        feat_spec,
                        oh_map,
                        ord_map,
                        num_ranges,
                    )

                    # Check if decoded outcome is valid
                    if decoded_outcome.empty:
                        scenario_failures += 1
                        continue

                    # Extract relevant outcome metrics and add patient ID
                    outcome_dict = decoded_outcome.to_dict("records")[0]
                    outcome_dict["_id"] = pid
                    outcome_dict["treatment_scenario"] = scenario_name
                    outcome_dict["episodes.surgery"] = treatment_vector[0]
                    outcome_dict["episodes.chemotherapy"] = treatment_vector[1]
                    outcome_dict["episodes.radiotherapy"] = treatment_vector[2]

                    # Extract endpoint probabilities
                    endpoint_prob_cols = [
                        "episodes.treatments.fields.endpoint_0",
                        "episodes.treatments.fields.endpoint_1",
                        "episodes.treatments.fields.endpoint_2",
                    ]

                    for prob_col in endpoint_prob_cols:
                        if prob_col in decoded_outcome.columns:
                            outcome_dict[prob_col] = decoded_outcome[prob_col].iloc[0]

                    # Compute most likely outcome if probabilities exist
                    if all(col in outcome_dict for col in endpoint_prob_cols):
                        probs = [outcome_dict[col] for col in endpoint_prob_cols]
                        most_likely_endpoint = np.argmax(probs) + 1
                        outcome_dict["episodes.treatments.fields.endpoint"] = (
                            most_likely_endpoint
                        )
                        outcome_dict["endpoint_max_probability"] = max(probs)

                        outcome_dict["local_recurrence_prob"] = outcome_dict.get(
                            "episodes.treatments.fields.endpoint_0", 0
                        )
                        outcome_dict["metastasis_prob"] = outcome_dict.get(
                            "episodes.treatments.fields.endpoint_1", 0
                        )
                        outcome_dict["death_prob"] = outcome_dict.get(
                            "episodes.treatments.fields.endpoint_2", 0
                        )

                    scenario_outcomes.append(outcome_dict)
                    scenario_successes += 1

                except Exception as e:
                    print(
                        f"Error processing patient {pid} for recurrence analysis: {e}"
                    )
                    scenario_failures += 1
                    continue

        # Add all scenario outcomes to main list
        counterfactual_data.extend(scenario_outcomes)
        total_successes += scenario_successes
        total_failures += scenario_failures

        print(
            f"Scenario {scenario_name} completed: {scenario_successes} successes, {scenario_failures} failures"
        )

    print(
        f"Total recurrence inference: {total_successes} successes, {total_failures} failures"
    )
    print(
        f"Recurrence success rate: {total_successes/(total_successes + total_failures)*100:.1f}%"
        if (total_successes + total_failures) > 0
        else "Recurrence success rate: 0%"
    )

    if counterfactual_data:
        generated_df = pd.DataFrame(counterfactual_data)
        print(
            f"Successfully generated {len(generated_df)} counterfactual recurrence outcomes"
        )

        # Save combined CSV file with all scenarios
        combined_csv_filename = "counterfactual_recurrence_outcomes_all_scenarios.csv"
        generated_df.to_csv(combined_csv_filename, index=False)
        print(
            f"Saved combined dataset with {len(generated_df)} records to {combined_csv_filename}"
        )

        # Display summary of endpoint probabilities if available
        endpoint_prob_cols = ["local_recurrence_prob", "metastasis_prob", "death_prob"]
        available_prob_cols = [
            col for col in endpoint_prob_cols if col in generated_df.columns
        ]

        if available_prob_cols:
            print(f"\nEndpoint Probability Summary by Treatment Scenario:")
            for scenario in generated_df["treatment_scenario"].unique():
                scenario_data = generated_df[
                    generated_df["treatment_scenario"] == scenario
                ]
                print(f"\n{scenario.upper()}:")
                for prob_col in available_prob_cols:
                    if prob_col in scenario_data.columns:
                        mean_prob = scenario_data[prob_col].mean()
                        std_prob = scenario_data[prob_col].std()
                        print(
                            f"  {prob_col.replace('_', ' ').title()}: {mean_prob:.4f} ± {std_prob:.4f}"
                        )

        return generated_df
    else:
        print(
            "Failed to generate counterfactual recurrence data, using original observational data"
        )
        return df


def compute_local_recurrence_rates(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute local recurrence rates for surgery vs surgery+RT groups."""
    results = {}

    # Check if endpoint field exists - look for both categorical and probability columns
    endpoint_col = "episodes.treatments.fields.endpoint"
    endpoint_prob_cols = [
        "episodes.treatments.fields.endpoint_0",  # Local recurrence probability (class 1)
        "episodes.treatments.fields.endpoint_1",  # Metastasis probability (class 2)
        "episodes.treatments.fields.endpoint_2",  # Death probability (class 3)
    ]

    # Check for categorical endpoint column
    if endpoint_col in df.columns:
        # Convert to float first, then to Int64 to handle string floats like '1.0'
        df[endpoint_col] = pd.to_numeric(df[endpoint_col], errors="coerce").astype(
            "Int64"
        )
        use_categorical = True
    else:
        use_categorical = False
        print(
            f"Warning: {endpoint_col} column not found. Looking for probability columns..."
        )

    # Check for probability columns (one-hot encoded)
    available_prob_cols = [col for col in endpoint_prob_cols if col in df.columns]
    use_probabilities = len(available_prob_cols) > 0

    if not use_categorical and not use_probabilities:
        # Try alternative column names
        alternative_cols = [
            col
            for col in df.columns
            if "endpoint" in col.lower() or "recurrence" in col.lower()
        ]
        if alternative_cols:
            endpoint_col = alternative_cols[0]
            print(f"Using alternative column: {endpoint_col}")
            use_categorical = True
        else:
            print("No endpoint/recurrence columns found")
            return {"error": "No endpoint data available"}

    # Define treatment groups
    surgery_only = df[
        (df["episodes.surgery"] == 1) & (df["episodes.radiotherapy"] == 0)
    ]

    surgery_rt = df[(df["episodes.surgery"] == 1) & (df["episodes.radiotherapy"] == 1)]

    if len(surgery_only) == 0 or len(surgery_rt) == 0:
        return {"error": "Insufficient patients in treatment groups"}

    # Calculate rates using probabilities if available
    if use_probabilities:
        print("Using probability distributions for endpoint analysis...")

        # Map probability columns to outcome names
        prob_outcome_mapping = {
            "episodes.treatments.fields.endpoint_0": "local_recurrence",
            "episodes.treatments.fields.endpoint_1": "metastasis",
            "episodes.treatments.fields.endpoint_2": "death",
        }

        # Extract probabilities for each outcome
        for prob_col, outcome_name in prob_outcome_mapping.items():
            if prob_col in df.columns:
                # Convert probabilities to numeric
                df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")

                # Calculate mean probabilities for each treatment group
                surgery_only_prob = (
                    surgery_only[prob_col].mean() if len(surgery_only) > 0 else 0
                )
                surgery_rt_prob = (
                    surgery_rt[prob_col].mean() if len(surgery_rt) > 0 else 0
                )

                # Calculate standard deviations
                surgery_only_std = (
                    surgery_only[prob_col].std() if len(surgery_only) > 0 else 0
                )
                surgery_rt_std = (
                    surgery_rt[prob_col].std() if len(surgery_rt) > 0 else 0
                )

                # Store probability results
                results[f"surgery_only_{outcome_name}_prob_mean"] = surgery_only_prob
                results[f"surgery_rt_{outcome_name}_prob_mean"] = surgery_rt_prob
                results[f"surgery_only_{outcome_name}_prob_std"] = surgery_only_std
                results[f"surgery_rt_{outcome_name}_prob_std"] = surgery_rt_std

                # Calculate absolute risk difference
                prob_difference = surgery_only_prob - surgery_rt_prob
                results[f"{outcome_name}_prob_difference"] = prob_difference

                # Calculate relative risk reduction
                if surgery_only_prob > 0:
                    relative_reduction = prob_difference / surgery_only_prob
                    results[f"{outcome_name}_relative_risk_reduction"] = (
                        relative_reduction
                    )
                else:
                    results[f"{outcome_name}_relative_risk_reduction"] = 0

                # Calculate number needed to treat
                if prob_difference > 0:
                    nnt = 1 / prob_difference
                    results[f"{outcome_name}_number_needed_to_treat"] = nnt
                else:
                    results[f"{outcome_name}_number_needed_to_treat"] = float("inf")

                print(f"\n{outcome_name.replace('_', ' ').title()} Probabilities:")
                print(f"Surgery Only: {surgery_only_prob:.4f} ± {surgery_only_std:.4f}")
                print(f"Surgery + RT: {surgery_rt_prob:.4f} ± {surgery_rt_std:.4f}")
                print(f"Absolute Risk Reduction: {prob_difference:.4f}")
                print(
                    f"Relative Risk Reduction: {results[f'{outcome_name}_relative_risk_reduction']:.4f}"
                )

                # Count patients based on probability thresholds (e.g., >0.5 for positive prediction)
                threshold = 0.5
                surgery_only_predicted = len(
                    surgery_only[surgery_only[prob_col] > threshold]
                )
                surgery_rt_predicted = len(surgery_rt[surgery_rt[prob_col] > threshold])

                results[f"surgery_only_{outcome_name}_predicted_count"] = (
                    surgery_only_predicted
                )
                results[f"surgery_rt_{outcome_name}_predicted_count"] = (
                    surgery_rt_predicted
                )

                surgery_only_predicted_rate = (
                    surgery_only_predicted / len(surgery_only)
                    if len(surgery_only) > 0
                    else 0
                )
                surgery_rt_predicted_rate = (
                    surgery_rt_predicted / len(surgery_rt) if len(surgery_rt) > 0 else 0
                )

                results[f"surgery_only_{outcome_name}_predicted_rate"] = (
                    surgery_only_predicted_rate
                )
                results[f"surgery_rt_{outcome_name}_predicted_rate"] = (
                    surgery_rt_predicted_rate
                )

                print(
                    f"Predicted cases (prob > {threshold}): Surgery={surgery_only_predicted}/{len(surgery_only)} ({surgery_only_predicted_rate:.3f}), Surgery+RT={surgery_rt_predicted}/{len(surgery_rt)} ({surgery_rt_predicted_rate:.3f})"
                )

        # Store method used
        results["analysis_method"] = "probabilities"

    # Also calculate categorical rates if available
    if use_categorical:
        print("Using categorical endpoint data for validation...")

        # Calculate local recurrence rates (endpoint = 1)
        surgery_only_recurrences = surgery_only[surgery_only[endpoint_col] == 1]
        surgery_rt_recurrences = surgery_rt[surgery_rt[endpoint_col] == 1]

        surgery_only_rate = (
            len(surgery_only_recurrences) / len(surgery_only)
            if len(surgery_only) > 0
            else 0
        )
        surgery_rt_rate = (
            len(surgery_rt_recurrences) / len(surgery_rt) if len(surgery_rt) > 0 else 0
        )

        results.update(
            {
                "surgery_only_recurrence_rate_categorical": surgery_only_rate,
                "surgery_rt_recurrence_rate_categorical": surgery_rt_rate,
                "surgery_only_recurrences_categorical": len(surgery_only_recurrences),
                "surgery_rt_recurrences_categorical": len(surgery_rt_recurrences),
                "absolute_risk_reduction_categorical": surgery_only_rate
                - surgery_rt_rate,
                "relative_risk_reduction_categorical": (
                    (surgery_only_rate - surgery_rt_rate) / surgery_only_rate
                    if surgery_only_rate > 0
                    else 0
                ),
                "number_needed_to_treat_categorical": (
                    1 / (surgery_only_rate - surgery_rt_rate)
                    if (surgery_only_rate - surgery_rt_rate) > 0
                    else float("inf")
                ),
            }
        )

        # Calculate rates for other endpoints
        for endpoint_val, endpoint_name in [(2, "metastasis"), (3, "death")]:
            surgery_only_events = len(
                surgery_only[surgery_only[endpoint_col] == endpoint_val]
            )
            surgery_rt_events = len(
                surgery_rt[surgery_rt[endpoint_col] == endpoint_val]
            )

            surgery_only_event_rate = (
                surgery_only_events / len(surgery_only) if len(surgery_only) > 0 else 0
            )
            surgery_rt_event_rate = (
                surgery_rt_events / len(surgery_rt) if len(surgery_rt) > 0 else 0
            )

            results[f"surgery_only_{endpoint_name}_rate_categorical"] = (
                surgery_only_event_rate
            )
            results[f"surgery_rt_{endpoint_name}_rate_categorical"] = (
                surgery_rt_event_rate
            )
            results[f"surgery_only_{endpoint_name}_count_categorical"] = (
                surgery_only_events
            )
            results[f"surgery_rt_{endpoint_name}_count_categorical"] = surgery_rt_events

        print(f"\nCategorical Local Recurrence Rates:")
        print(
            f"Surgery Only: {surgery_only_rate:.3f} ({len(surgery_only_recurrences)}/{len(surgery_only)})"
        )
        print(
            f"Surgery + RT: {surgery_rt_rate:.3f} ({len(surgery_rt_recurrences)}/{len(surgery_rt)})"
        )
        print(
            f"Absolute Risk Reduction: {results['absolute_risk_reduction_categorical']:.3f}"
        )
        print(
            f"Relative Risk Reduction: {results['relative_risk_reduction_categorical']:.3f}"
        )

        if not use_probabilities:
            results["analysis_method"] = "categorical"
        else:
            results["analysis_method"] = "both"

    return results


def compare_recurrence_statistically(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform statistical tests comparing recurrence rates between groups."""
    results = {}

    endpoint_col = "episodes.treatments.fields.endpoint"
    if endpoint_col not in df.columns:
        alternative_cols = [col for col in df.columns if "endpoint" in col.lower()]
        if alternative_cols:
            endpoint_col = alternative_cols[0]

        else:
            return {"error": "No endpoint data for statistical testing"}

    # Define treatment groups
    surgery_only = df[
        (df["episodes.surgery"] == 1) & (df["episodes.radiotherapy"] == 0)
    ]

    surgery_rt = df[(df["episodes.surgery"] == 1) & (df["episodes.radiotherapy"] == 1)]

    if len(surgery_only) == 0 or len(surgery_rt) == 0:
        return {"error": "Insufficient data for statistical testing"}

    # Create contingency table for local recurrence (endpoint = 1)
    surgery_only_recur = len(surgery_only[surgery_only[endpoint_col] == 1])
    surgery_only_no_recur = len(surgery_only) - surgery_only_recur
    surgery_rt_recur = len(surgery_rt[surgery_rt[endpoint_col] == 1])
    surgery_rt_no_recur = len(surgery_rt) - surgery_rt_recur

    contingency_table = np.array(
        [
            [surgery_only_recur, surgery_only_no_recur],
            [surgery_rt_recur, surgery_rt_no_recur],
        ]
    )

    # Chi-square test
    try:
        chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
        results["chi2_statistic"] = chi2
        results["chi2_p_value"] = p_chi2
    except Exception as e:
        print(f"Chi-square test failed: {e}")
        results["chi2_p_value"] = float("nan")

    # Fisher's exact test (more appropriate for small samples)
    try:
        odds_ratio, p_fisher = fisher_exact(contingency_table)
        results["fisher_p_value"] = p_fisher
        results["odds_ratio"] = odds_ratio
    except Exception as e:
        print(f"Fisher's exact test failed: {e}")
        results["fisher_p_value"] = float("nan")
        results["odds_ratio"] = float("nan")

    # Risk ratio
    surgery_only_rate = (
        surgery_only_recur / len(surgery_only) if len(surgery_only) > 0 else 0
    )
    surgery_rt_rate = surgery_rt_recur / len(surgery_rt) if len(surgery_rt) > 0 else 0

    if surgery_rt_rate > 0:
        risk_ratio = surgery_only_rate / surgery_rt_rate
        results["risk_ratio"] = risk_ratio
    else:
        results["risk_ratio"] = float("inf") if surgery_only_rate > 0 else float("nan")

    print(f"\nStatistical Tests:")
    if "fisher_p_value" in results:
        print(f"Fisher's exact test p-value: {results['fisher_p_value']:.4f}")
    if "odds_ratio" in results:
        print(f"Odds ratio: {results['odds_ratio']:.3f}")
    if "risk_ratio" in results:
        print(f"Risk ratio: {results['risk_ratio']:.3f}")

    return results


def plot_local_recurrence_analysis(df: pd.DataFrame, filename: str) -> None:
    """Create visualization of local recurrence analysis."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        endpoint_col = "episodes.treatments.fields.endpoint"
        df[endpoint_col] = df.get(
            endpoint_col, pd.Series(dtype="int")
        )  # Ensure column exists
        if endpoint_col not in df.columns:
            alternative_cols = [col for col in df.columns if "endpoint" in col.lower()]
            if alternative_cols:
                endpoint_col = alternative_cols[0]
            else:
                print("Cannot create recurrence plot: no endpoint data")
                return

        # Define treatment groups
        surgery_only = df[
            (df["episodes.surgery"] == 1) & (df["episodes.radiotherapy"] == 0)
        ]

        surgery_rt = df[
            (df["episodes.surgery"] == 1) & (df["episodes.radiotherapy"] == 1)
        ]

        if len(surgery_only) == 0 or len(surgery_rt) == 0:
            print("Cannot create recurrence plot: insufficient data")
            return

        # Plot 1: Recurrence rates by treatment
        endpoints = [1, 2, 3]
        endpoint_names = ["Local Recurrence", "Metastasis", "Death"]
        colors = ["red", "orange", "black"]

        surgery_only_rates = []
        surgery_rt_rates = []

        for endpoint in endpoints:
            surgery_only_count = len(
                surgery_only[surgery_only[endpoint_col] == endpoint]
            )
            surgery_rt_count = len(surgery_rt[surgery_rt[endpoint_col] == endpoint])

            surgery_only_rate = (
                surgery_only_count / len(surgery_only) if len(surgery_only) > 0 else 0
            )
            surgery_rt_rate = (
                surgery_rt_count / len(surgery_rt) if len(surgery_rt) > 0 else 0
            )

            surgery_only_rates.append(surgery_only_rate)
            surgery_rt_rates.append(surgery_rt_rate)

        x = np.arange(len(endpoint_names))
        width = 0.35

        bars1 = axes[0, 0].bar(
            x - width / 2,
            surgery_only_rates,
            width,
            label="Surgery Only",
            alpha=0.8,
            color="lightblue",
        )
        bars2 = axes[0, 0].bar(
            x + width / 2,
            surgery_rt_rates,
            width,
            label="Surgery + RT",
            alpha=0.8,
            color="lightcoral",
        )

        axes[0, 0].set_xlabel("Endpoint")
        axes[0, 0].set_ylabel("Rate")
        axes[0, 0].set_title("Event Rates by Treatment Group")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(endpoint_names)
        axes[0, 0].legend()

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Plot 2: Local recurrence comparison focus
        local_recur_data = [
            len(surgery_only[surgery_only[endpoint_col] == 1]),
            len(surgery_only) - len(surgery_only[surgery_only[endpoint_col] == 1]),
            len(surgery_rt[surgery_rt[endpoint_col] == 1]),
            len(surgery_rt) - len(surgery_rt[surgery_rt[endpoint_col] == 1]),
        ]

        labels = [
            "S: Recurrence",
            "S: No Recurrence",
            "S+RT: Recurrence",
            "S+RT: No Recurrence",
        ]
        colors_pie = ["lightcoral", "lightblue", "red", "blue"]

        axes[0, 1].pie(
            local_recur_data,
            labels=labels,
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0, 1].set_title("Local Recurrence Distribution")

        # Plot 3: Risk metrics
        surgery_only_recur_rate = (
            len(surgery_only[surgery_only[endpoint_col] == 1]) / len(surgery_only)
            if len(surgery_only) > 0
            else 0
        )
        surgery_rt_recur_rate = (
            len(surgery_rt[surgery_rt[endpoint_col] == 1]) / len(surgery_rt)
            if len(surgery_rt) > 0
            else 0
        )

        arr = surgery_only_recur_rate - surgery_rt_recur_rate  # Absolute risk reduction
        rrr = (
            arr / surgery_only_recur_rate if surgery_only_recur_rate > 0 else 0
        )  # Relative risk reduction
        nnt = 1 / arr if arr > 0 else float("inf")  # Number needed to treat

        metrics = [
            "Absolute Risk\nReduction",
            "Relative Risk\nReduction",
            "Number Needed\nto Treat",
        ]
        values = [arr, rrr, min(nnt, 100)]  # Cap NNT for visualization

        bars3 = axes[1, 0].bar(
            metrics, values, color=["green", "blue", "purple"], alpha=0.7
        )
        axes[1, 0].set_title("Treatment Effect Metrics")
        axes[1, 0].set_ylabel("Value")

        # Add value labels
        for bar, val in zip(bars3, [arr, rrr, nnt]):
            label = f"{val:.3f}" if val != float("inf") else "∞"
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Plot 4: Sample sizes
        sample_data = [len(surgery_only), len(surgery_rt)]
        sample_labels = ["Surgery Only", "Surgery + RT"]

        bars4 = axes[1, 1].bar(
            sample_labels, sample_data, color=["lightblue", "lightcoral"], alpha=0.8
        )
        axes[1, 1].set_title("Sample Sizes")
        axes[1, 1].set_ylabel("Number of Patients")

        # Add value labels
        for bar, val in zip(bars4, sample_data):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.suptitle(
            "Local Recurrence Analysis: Surgery vs Surgery + Radiotherapy",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Local recurrence analysis plot saved: {filename}")

    except Exception as exc:
        print(f"Local recurrence plot creation failed: {exc}")


def analyze_time_to_recurrence(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze time to local recurrence using survival analysis methods."""
    results = {}

    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        endpoint_col = "episodes.treatments.fields.endpoint"
        if endpoint_col not in df.columns:
            alternative_cols = [col for col in df.columns if "endpoint" in col.lower()]
            if alternative_cols:
                endpoint_col = alternative_cols[0]
            else:
                return {"error": "No endpoint data for time-to-event analysis"}

        # Try to find time columns
        time_col = None
        for col in df.columns:
            if "time" in col.lower() and "difference" in col.lower():
                time_col = col
                break

        if time_col is None:
            print("No time column found for time-to-recurrence analysis")
            return {"error": "No time data available"}

        # Define treatment groups
        surgery_only = df[
            (df["episodes.surgery"] == 1) & (df["episodes.radiotherapy"] == 0)
        ]

        surgery_rt = df[
            (df["episodes.surgery"] == 1) & (df["episodes.radiotherapy"] == 1)
        ]

        if len(surgery_only) == 0 or len(surgery_rt) == 0:
            return {"error": "Insufficient data for time-to-event analysis"}

        # Kaplan-Meier analysis
        kmf_surgery = KaplanMeierFitter()
        kmf_surgery_rt = KaplanMeierFitter()

        kmf_surgery.fit(
            surgery_only[time_col],
            surgery_only["recurrence_event"],
            label="Surgery Only",
        )
        kmf_surgery_rt.fit(
            surgery_rt[time_col], surgery_rt["recurrence_event"], label="Surgery + RT"
        )

        # Log-rank test
        logrank_result = logrank_test(
            surgery_only[time_col],
            surgery_rt[time_col],
            surgery_only["recurrence_event"],
            surgery_rt["recurrence_event"],
        )

        results.update(
            {
                "logrank_p_value": logrank_result.p_value,
                "logrank_test_statistic": logrank_result.test_statistic,
                "median_time_to_recurrence_surgery": kmf_surgery.median_survival_time_,
                "median_time_to_recurrence_surgery_rt": kmf_surgery_rt.median_survival_time_,
            }
        )

        # Plot Kaplan-Meier curves
        plt.figure(figsize=(10, 6))
        ax = kmf_surgery.plot(color="blue", ci_show=True)
        kmf_surgery_rt.plot(ax=ax, color="red", ci_show=True)

        plt.title("Time to Local Recurrence: Surgery vs Surgery + Radiotherapy")
        plt.xlabel("Time (years)")
        plt.ylabel("Recurrence-Free Survival Probability")
        plt.grid(True, alpha=0.3)

        # Add p-value annotation
        plt.text(
            0.05,
            0.2,
            f"Log-rank p = {logrank_result.p_value:.4f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig("time_to_recurrence_km.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Time-to-recurrence Kaplan-Meier plot saved: time_to_recurrence_km.png")

        print(f"\nTime-to-Recurrence Analysis:")
        print(f"Log-rank test p-value: {logrank_result.p_value:.4f}")
        print(
            f"Median time to recurrence (Surgery): {results['median_time_to_recurrence_surgery']:.2f} years"
            if results["median_time_to_recurrence_surgery"]
            else "Median not reached (Surgery)"
        )
        print(
            f"Median time to recurrence (Surgery+RT): {results['median_time_to_recurrence_surgery_rt']:.2f} years"
            if results["median_time_to_recurrence_surgery_rt"]
            else "Median not reached (Surgery+RT)"
        )

    except Exception as e:
        print(f"Time-to-recurrence analysis failed: {e}")
        results["error"] = f"Time-to-recurrence analysis failed: {e}"

    return results


def decode_embedding_endpoint(
    embedding: torch.Tensor,
    feat_spec: Dict,
    oh_map: Dict,
    ord_map: Dict,
    num_ranges: Dict,
) -> pd.DataFrame:
    """
    Decode embedding tensor specifically for endpoint analysis, preserving probability columns.

    This function is similar to decode_embedding but specifically preserves the endpoint
    probability columns without aggregating them into a single categorical value.

    Args:
        embedding: Tensor to decode
        feat_spec: Feature specifications
        oh_map: One-hot mappings
        ord_map: Ordinal mappings
        num_ranges: Numeric ranges

    Returns:
        DataFrame with decoded features including individual endpoint probabilities
    """
    try:
        # Import the original decode_embedding function
        from Models.models.decoder_embedders import decode_embedding

        # First get the standard decoded output
        decoded_df = decode_embedding(
            embedding, feat_spec, oh_map, ord_map, num_ranges, one_hot_argmax=False
        )

        if decoded_df.empty:
            return decoded_df

        # Check if endpoint columns exist in the decoded output
        endpoint_cols = [
            "episodes.treatments.fields.endpoint_1.0",  # Local recurrence
            "episodes.treatments.fields.endpoint_2.0",  # Metastasis
            "episodes.treatments.fields.endpoint_3.0",  # Death
        ]

        # Preserve the individual endpoint probability columns
        endpoint_data = {}
        for col in endpoint_cols:
            if col in decoded_df.columns:
                endpoint_data[col] = (
                    decoded_df[col].iloc[0] if len(decoded_df) > 0 else 0.0
                )

        # If we have endpoint probabilities, compute additional metrics
        if len(endpoint_data) == 3:  # All three endpoint probabilities exist
            probs = list(endpoint_data.values())

            # Compute the most likely endpoint (1-indexed)
            most_likely_endpoint = np.argmax(probs) + 1
            max_probability = max(probs)

            # Add computed categorical endpoint
            endpoint_data["episodes.treatments.fields.endpoint"] = most_likely_endpoint
            endpoint_data["endpoint_max_probability"] = max_probability

            # Add cleaner named probability columns
            endpoint_data["local_recurrence_prob"] = endpoint_data[
                "episodes.treatments.fields.endpoint_1.0"
            ]
            endpoint_data["metastasis_prob"] = endpoint_data[
                "episodes.treatments.fields.endpoint_2.0"
            ]
            endpoint_data["death_prob"] = endpoint_data[
                "episodes.treatments.fields.endpoint_3.0"
            ]

            # Add probability confidence metrics
            endpoint_data["endpoint_entropy"] = -sum(
                p * np.log(p + 1e-8) for p in probs if p > 0
            )
            endpoint_data["endpoint_confidence"] = (
                max_probability - (sum(probs) - max_probability) / 2
            )

        # Create a new dataframe that preserves all original columns plus endpoint details
        result_df = decoded_df.copy()

        # Update with our endpoint analysis
        for key, value in endpoint_data.items():
            result_df[key] = value

        return result_df

    except Exception as e:
        print(f"Error in decode_embedding_endpoint: {e}")
        # Fallback to original decode_embedding if there's an error
        try:
            from Models.models.decoder_embedders import decode_embedding

            return decode_embedding(embedding, feat_spec, oh_map, ord_map, num_ranges)
        except Exception as fallback_error:
            print(f"Fallback decode_embedding also failed: {fallback_error}")
            return pd.DataFrame()