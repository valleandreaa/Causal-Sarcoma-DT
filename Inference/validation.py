from scipy.stats import ttest_ind
from scipy.stats import wasserstein_distance, ks_2samp, entropy
from Inference.analysis import load_and_prepare
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple
import torch

# Placeholder for your model and loss function
def get_model_and_loss():
    """
    This is a placeholder function.
    Replace this with your actual model loading and loss function definition.
    """
    # Example: a simple linear model
    model = torch.nn.Linear(10, 1) 
    # Example: Mean Squared Error loss
    loss_fn = torch.nn.MSELoss()
    return model, loss_fn

# ------------------------------------------------------------------ #
# Validation constants
# ------------------------------------------------------------------ #
REAL_OUTCOME_COL = "real_episodes.diagnosis.fields.status"
FAKE_OUTCOME_PROB_COL = "fake_episodes.diagnosis.fields.status_AWD_prob"
# FAKE_OUTCOME_COL = "fake_episodes.treatments.fields.endpoint_1.0_prob"
HAS_CLIN_COL     = "has_clinical_data"
SURGERY_FLAG_COL     = "surgery"
RADIOTHERAPY_FLAG_COL= "radiotherapy"
CHEMOTHERAPY_FLAG_COL= "chemotherapy"
TIMESTEP_COL         = "timestep"
HISTOLOGICAL_DIAGNOSIS_COL = "tumor_characteristics.histological_diagnosis"
# csv_path = "counterfactual_scenarios_multiple_patients_no_embed.csv"
csv_path = "counterfactual_scenarios_validation_patients.csv"
def evaluate_physiological_protocol_feasibility(df: pd.DataFrame) -> float:
    """
    1) any negative numeric values (e.g. tumor volumes < 0)
    2) chemo or RT without surgery (violates basic sarcoma guideline)
    Returns fraction of all rows that fail.
    """
    # 1) negative‐value check
    numeric = df.select_dtypes(include=[np.number])
    neg_violation = (numeric < 0).any(axis=1)

    # 2) simple protocol: no chemo/RT without surgery
    proto_violation = (
        (df[CHEMOTHERAPY_FLAG_COL] == 1) |
        (df[RADIOTHERAPY_FLAG_COL] == 1)
    ) & (df[SURGERY_FLAG_COL] == 0)
    
    tumor_violation = (
        (df["fake_episodes.treatments.fields.tumor_max_size"] < 0) &
        (df[SURGERY_FLAG_COL] == 1)
    )

    violations = neg_violation | proto_violation | tumor_violation
    frac_fail = violations.mean()
    print(f"Physiol./protocol failures: {violations.sum()}/{len(df)} ({frac_fail:.2%})")
    return frac_fail

def plot_distributions(
    df: pd.DataFrame,
    features: list,
    out_path: str = "distributions.png",
    bins: int = 30
) -> None:
    """
    Overlayed histogram of real vs. fake features.
    Creates a 2x2 grid of plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        
        plot_df = df
        if feature in ["radiotherapy_total_dose", "redonc_fractions"]:
            plot_df = df[df[RADIOTHERAPY_FLAG_COL] == 1].copy()

        if feature == "prom_score":
            real_col = f"real_general.{feature}"
            fake_col = f"fake_general.{feature}"
        else:
            real_col = f"real_episodes.treatments.fields.{feature}"
            fake_col = f"fake_episodes.treatments.fields.{feature}"

        real_vals = plot_df[real_col].dropna()
        fake_vals = plot_df[fake_col].dropna()

        if real_vals.empty and fake_vals.empty:
            ax.text(0.5, 0.5, f"No data for\n{feature}", ha='center', va='center')
            ax.set_title(f"Distribution of {feature.replace('_', ' ').title()}")
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel("Count")
            continue

        # histogram for fake
        if not fake_vals.empty:
            ax.hist(
                fake_vals,
                bins=bins,
                alpha=0.6,
                label="Fake",
                edgecolor="black"
            )
        # histogram for real
        if not real_vals.empty:
            ax.hist(
                real_vals,
                bins=bins,
                alpha=0.6,
                label="Real",
                edgecolor="black"
            )

        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Distributions plot saved to {out_path}")

def evaluate_natural_experiment(
    df: pd.DataFrame,
    match_cols
) -> Tuple[float, float]:
    """
    Compares real outcomes (binary) with fake outcome probabilities.
    - Real outcomes are mapped to 0/1 ('DOD' -> 1, others -> 0).
    - Fake outcomes are the predicted probabilities of 'DOD'.
    - Wasserstein-1 distance is the absolute difference in means.
    - A two-sample Welch's t-test is used for statistical comparison.
    """
    real_df = df[df[HAS_CLIN_COL] == True].copy()
    fake_df = df.copy() # Use all data for fake predictions

    # Binarize real outcome: 1 if 'DOD', 0 otherwise
    y_real = (real_df[REAL_OUTCOME_COL] == 'DOD').astype(int)
    
    # Use the predicted probability of 'DOD' for fake outcomes
    y_fake = fake_df[FAKE_OUTCOME_PROB_COL]

    # Wasserstein-1 distance = absolute difference in means
    wd = np.abs(y_real.mean() - y_fake.mean())
    
    # Two-sample Welch's t-test
    _, t_p = ttest_ind(y_real.dropna(), y_fake.dropna(), equal_var=False)
    
    print(f"Natural experiment → W₁={wd:.4f}, T-test p={t_p:.4g}")
    return wd, t_p


def evaluate_marginal_distribution_alignment(
    df: pd.DataFrame,
    treatment_col: str,
    num_bins: int = 20
) -> float:
    """
    For each treatment in real data, compute KL( P_fake(Y|T) || P_real(Y|T) ).
    This version discretizes the fake probabilities to make them comparable to real outcomes.
    Returns weighted KL divergence.
    """
    real_df = df[df[HAS_CLIN_COL] == True].copy()
    fake_df = df.copy()

    # Determine discrete fake outcomes from probabilities
    prob_cols = [
        'fake_episodes.diagnosis.fields.status_AWD_prob',
        'fake_episodes.diagnosis.fields.status_DOD_prob',
        'fake_episodes.diagnosis.fields.status_NED_prob'
    ]
    # Get the name of the outcome with the highest probability
    fake_df['fake_outcome'] = fake_df[prob_cols].idxmax(axis=1).str.split('_').str[-2]

    # P_real(T)
    pT = real_df[treatment_col].value_counts(normalize=True)

    kl_total = 0.0
    all_outcomes = sorted(df[REAL_OUTCOME_COL].dropna().unique())

    for t, p_t in pT.items():
        # Get distributions of outcomes for the given treatment
        real_dist = real_df.loc[real_df[treatment_col] == t, REAL_OUTCOME_COL].value_counts(normalize=True)
        fake_dist = fake_df.loc[fake_df[treatment_col] == t, 'fake_outcome'].value_counts(normalize=True)

        # Align distributions to have the same outcome categories
        real_dist = real_dist.reindex(all_outcomes, fill_value=0)
        fake_dist = fake_dist.reindex(all_outcomes, fill_value=0)

        if real_dist.sum() == 0 or fake_dist.sum() == 0:
            continue

        # Smooth distributions
        eps = 1e-10
        pr = real_dist.values + eps
        pf = fake_dist.values + eps
        pr /= pr.sum()
        pf /= pf.sum()

        # Calculate KL divergence for this treatment and weight it by P(T)
        kl_t = entropy(pf, pr) # KL(P_fake || P_real)
        kl_total += p_t * kl_t

    print(f"Marginal KL divergence: {kl_total:.4f}")
    return kl_total


def evaluate_joint_distribution_alignment(df: pd.DataFrame, treatment_col: str) -> float:
    """
    Computes KL divergence between the joint distribution of synthetic (treatment, outcome)
    and the empirical distribution from real data: KL( P_real(T,Y) || P_fake(T,Y) ).
    A large shift would flag poor calibration.
    """
    real_df = df.copy()
    fake_df = df.copy() # Counterfactuals are generated for all patients

    # 1. Determine outcomes for real and fake data
    # Real outcomes are in REAL_OUTCOME_COL
    real_df['outcome'] = real_df[REAL_OUTCOME_COL]

    # Fake outcomes: determine from probabilities
    prob_cols = [
        'fake_episodes.diagnosis.fields.status_AWD_prob',
        'fake_episodes.diagnosis.fields.status_DOD_prob',
        'fake_episodes.diagnosis.fields.status_NED_prob'
    ]
    outcome_labels = [col.split('_')[-2] for col in prob_cols]
    fake_df['outcome'] = fake_df[prob_cols].idxmax(axis=1).str.split('_').str[-2]

    # 2. Compute joint probability distributions P(T, Y)
    real_joint_dist = real_df.groupby([treatment_col, 'outcome']).size().unstack(fill_value=0)
    fake_joint_dist = fake_df.groupby([treatment_col, 'outcome']).size().unstack(fill_value=0)

    # Align columns (outcomes) to ensure they match
    all_outcomes = sorted(list(set(real_joint_dist.columns) | set(fake_joint_dist.columns)))
    real_joint_dist = real_joint_dist.reindex(columns=all_outcomes, fill_value=0)
    fake_joint_dist = fake_joint_dist.reindex(columns=all_outcomes, fill_value=0)

    # Align rows (treatments)
    all_treatments = sorted(list(set(real_joint_dist.index) | set(fake_joint_dist.index)))
    real_joint_dist = real_joint_dist.reindex(index=all_treatments, fill_value=0)
    fake_joint_dist = fake_joint_dist.reindex(index=all_treatments, fill_value=0)

    # Normalize to get probabilities
    p_real = real_joint_dist.values.flatten() / real_joint_dist.values.sum()
    p_fake = fake_joint_dist.values.flatten() / fake_joint_dist.values.sum()

    # 3. Compute KL Divergence
    # Add a small epsilon to avoid division by zero or log(0)
    eps = 1e-10
    p_real_smooth = p_real + eps
    p_fake_smooth = p_fake + eps
    p_real_smooth /= p_real_smooth.sum()
    p_fake_smooth /= p_fake_smooth.sum()
    
    kl_div = entropy(p_real_smooth, p_fake_smooth)
    
    print(f"\nJoint Distribution KL Divergence: {kl_div:.4f}")
    return kl_div

def evaluate_influence_function_risk(df: pd.DataFrame, treatment_scenario: str) -> float:
    """
    Computes the Influence-Function Risk Estimation for a specific treatment scenario.
    """
    scenario_df = df[df['treatment_scenario'] == treatment_scenario].copy()
    
    if scenario_df.empty:
        print(f"No data for scenario '{treatment_scenario}', skipping IF risk calculation.")
        return np.nan

    model, loss_fn = get_model_and_loss()
    
    # This is a placeholder for feature columns
    # You should replace this with the actual feature columns used by your model
    feature_cols = [col for col in scenario_df.columns if scenario_df[col].dtype in [np.float64, np.int64] and "prob" not in col and "outcome" not in col and "status" not in col]
    
    # A simple way to handle categorical features is to one-hot encode them,
    # but for this example, we'll stick to numeric features.
    
    if not feature_cols:
        print("No numeric feature columns found for influence function calculation.")
        return 0.0

    if len(feature_cols) > 10:
        feature_cols = feature_cols[:10]
    elif len(feature_cols) < 10:
        print(f"Warning: Model expects 10 features, but only {len(feature_cols)} were found. This will fail.")
        # Pad with zeros if not enough features - this is a hack for the example to run
        # In a real scenario, you must ensure the data matches the model's input dimensions.
        num_missing_features = 10 - len(feature_cols)
        for i in range(num_missing_features):
            feature_name = f"missing_feature_{i}"
            scenario_df[feature_name] = 0
            feature_cols.append(feature_name)

    # Convert dataframe to tensor
    z_tensor = torch.tensor(scenario_df[feature_cols].fillna(0).values, dtype=torch.float32)
    
    # This is a placeholder for target column
    target_col = REAL_OUTCOME_COL 
    y_tensor = torch.tensor((scenario_df[target_col] == 'DOD').astype(int).values, dtype=torch.float32)

    # 1. Empirical Risk
    y_pred = model(z_tensor)
    empirical_risk = loss_fn(y_pred, y_tensor.view_as(y_pred))

    # 2. Influence Function part
    # We need per-sample gradients. A simple loop can achieve this.
    n = len(scenario_df)
    if n == 0:
        print(f"No samples for scenario '{treatment_scenario}' to calculate IF risk.")
        return np.nan
        
    grads = []
    for i in range(n):
        model.zero_grad()
        z_i = z_tensor[i:i+1]
        y_i = y_tensor[i:i+1]
        
        y_pred_i = model(z_i)
        loss = loss_fn(y_pred_i, y_i.view_as(y_pred_i))
        
        # We need the gradient of the loss with respect to the *parameters*, not the input z.
        # The formula uses \nabla_{\theta} \ell(\theta, z_i)
        grad_params = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
        
        # For simplicity, we'll flatten and concatenate the gradients of all parameters
        flat_grad = torch.cat([g.flatten() for g in grad_params if g is not None])
        if flat_grad.numel() > 0:
            grads.append(flat_grad)

    # This part of the formula seems to involve \nabla_z, which is unusual.
    # The standard IF for risk estimation involves gradients w.r.t. model parameters.
    # Let's assume the formula in the paper implies a simplification or a different context.
    # Re-implementing based on the provided formula: \nabla_z \ell(\theta; z)
    
    grads_z = []
    for i in range(n):
        model.zero_grad()
        z_i = z_tensor[i:i+1].clone().detach().requires_grad_(True)
        y_i = y_tensor[i:i+1]
        
        y_pred_i = model(z_i)
        loss = loss_fn(y_pred_i, y_i.view_as(y_pred_i))
        
        grad_z = torch.autograd.grad(loss, z_i, allow_unused=True)[0]
        if grad_z is not None:
            grads_z.append(grad_z)

    if not grads_z:
        print(f"Could not compute gradients for scenario '{treatment_scenario}'.")
        return np.nan

    grads_z_tensor = torch.stack(grads_z)
    avg_grad_z = grads_z_tensor.mean(dim=0)
    
    mu_hat = z_tensor.mean(dim=0)

    influence_values = torch.zeros(n)
    for i in range(n):
        z_i = z_tensor[i]
        influence_values[i] = torch.dot(avg_grad_z.squeeze(), z_i - mu_hat)

    # 3. Final IF Risk
    if_risk = empirical_risk + influence_values.mean()
    
    print(f"Influence-Function Risk: {if_risk.item():.4f}")
    
    # Non-parametric bootstrap for confidence intervals
    # This is computationally intensive and is simplified here.
    # A full implementation would re-calculate the influence values for each bootstrap sample.
    bootstrap_samples = 100
    risks = []
    for _ in range(bootstrap_samples):
        indices = np.random.choice(n, n, replace=True)
        risks.append(empirical_risk.item() + influence_values[indices].mean().item())

    lower_bound = np.percentile(risks, 2.5)
    upper_bound = np.percentile(risks, 97.5)

    print(f"95% CI for IF Risk: [{lower_bound:.4f}, {upper_bound:.4f}]")

    # --- Interpretation Script ---
    print("\n--- Influence-Function Risk Interpretation ---")
    
    # 1. Point Estimate
    print(f"1. Point Estimate: {if_risk.item():.2f}")
    print("   This is the best estimate of the average per-patient loss under the counterfactual distribution, corrected for selection bias.")

    # 2. Precision and Confidence Interval (CI)
    ci_width = upper_bound - lower_bound
    ci_width_percent = (ci_width / if_risk.item()) * 100 if if_risk.item() != 0 else 0
    std_error = ci_width / (2 * 1.96)
    
    print(f"\n2. Precision and CI:")
    print(f"   - The 95% CI spans {ci_width:.2f} units, which is {ci_width_percent:.1f}% of the point estimate.")
    print(f"   - Approximate Standard Error (SE): {std_error:.2f}. This suggests the estimator is reasonably precise.")

    # 3. Bias Correction
    risk_shift = if_risk.item() - empirical_risk.item()
    print(f"\n3. Bias Correction Magnitude:")
    print(f"   - Naive Empirical Risk: {empirical_risk.item():.2f}")
    print(f"   - IF-Corrected Risk:    {if_risk.item():.2f}")
    print(f"   - The IF term shifted the risk by {risk_shift:+.2f}, quantifying the degree of selection bias corrected.")
    print("--------------------------------------------\n")

    return if_risk.item()


def evaluate_natural_experiment_by_treatment(df: pd.DataFrame, match_cols: list, scenario_map: dict):
    """
    Performs a natural experiment by comparing real outcomes for a treatment scenario
    with counterfactual outcomes for a different, matched patient group.

    For each (real_scenario, fake_scenario) pair in scenario_map:
    1.  Identifies real patients who actually received the `real_scenario` treatment.
    2.  Identifies patients for whom we generated a `fake_scenario` counterfactual.
    3.  Matches these two groups based on `match_cols`.
    4.  Compares the real outcomes of the first group with the fake (predicted) outcomes
        of the second group using Wasserstein-1 distance and Welch's t-test.
    """
    print("\n--- Evaluating Natural Experiments by Treatment Scenario ---")
    
    for real_scenario, fake_scenario in scenario_map.items():
        print(f"\nComparing Real '{real_scenario}' vs. Fake '{fake_scenario}'")

        # Map treatment scenarios to binary flags
        def get_treatment_flags(scenario):
            if scenario == "S":
                return {"surgery": 1, "chemotherapy": 0, "radiotherapy": 0}
            elif scenario == "S_CT":
                return {"surgery": 1, "chemotherapy": 1, "radiotherapy": 0}
            elif scenario == "S_RT":
                return {"surgery": 1, "chemotherapy": 0, "radiotherapy": 1}
            elif scenario == "S_RT_CT":
                return {"surgery": 1, "chemotherapy": 1, "radiotherapy": 1}
            else:
                return None

        real_flags = get_treatment_flags(real_scenario)
        
        if real_flags is None:
            print(f"Unknown real scenario: {real_scenario}. Skipping.")
            continue

        # Group 1: Real data for patients who actually had the `real_scenario` treatment pattern
        real_condition = (df[HAS_CLIN_COL] == True)
        for flag, value in real_flags.items():
            # Column names for real data are like "real_episodes.treatments.fields.surgery"
            real_condition &= (df[f"actual_{flag}"] == value)
        real_group = df[real_condition].copy()
        
        # Group 2: Counterfactual data for patients under the `fake_scenario`
        fake_group = df[df['treatment_scenario'] == fake_scenario].copy()

        if real_group.empty:
            print(f"Warning: No real data found for scenario '{real_scenario}'. Skipping.")
            continue
        if fake_group.empty:
            print(f"Warning: No fake data found for scenario '{fake_scenario}'. Skipping.")
            continue

        # Prepare for merging
        real_group['outcome_binary'] = (real_group[REAL_OUTCOME_COL] == 'AWD').astype(int)
        real_outcomes = real_group[match_cols + ['outcome_binary']].dropna(subset=['outcome_binary'])
        
        fake_outcomes = fake_group[match_cols + [FAKE_OUTCOME_PROB_COL]].dropna(subset=[FAKE_OUTCOME_PROB_COL])

        # Find common patients based on matching columns
        merged = pd.merge(real_outcomes, fake_outcomes, on=match_cols, how='inner')

        if merged.empty:
            print(f"No matched patients found between real '{real_scenario}' and fake '{fake_scenario}'. Skipping.")
            continue
            
        y_real = merged['outcome_binary']
        y_fake = merged[FAKE_OUTCOME_PROB_COL]

        if len(y_real) < 2 or len(y_fake) < 2:
            print(f"Not enough matched samples to compare ({len(y_real)} real, {len(y_fake)} fake). Skipping.")
            continue

        # Calculate metrics
        wd = np.abs(y_real.mean() - y_fake.mean())
        _, t_p = ttest_ind(y_real, y_fake, equal_var=False)

        print(f"Found {len(merged)} matched patients.")
        print(f"Wasserstein-1 Distance: {wd:.4f}")
        print(f"T-test p-value: {t_p:.4g}")


def main():
    df = load_and_prepare(csv_path)

    # 1) feasibility
    fail_rate = evaluate_physiological_protocol_feasibility(df)
    
    features_to_plot = [
        "tumor_max_size",
        "radiotherapy_total_dose",
        "redonc_fractions",
        "prom_score"
    ]
    plot_distributions(df, features=features_to_plot, out_path="imgs/distributions.png")

    # 2) Joint distribution alignment
    evaluate_joint_distribution_alignment(
        df,
        treatment_col="treatment_scenario"
    )

    # 3) natural experiments (match on age + histo dx)
    wd, ks_p = evaluate_natural_experiment(
        df,
        match_cols=["general.age", "tumor_characteristics.histological_diagnosis"]
    )

    # 4) marginal alignment over your treatment scenarios
    kl = evaluate_marginal_distribution_alignment(
        df,
        treatment_col="treatment_scenario",
        num_bins=30
    )

    # 5) natural experiments by treatment scenario
    scenario_map = {
        "S_CT": "S_CT",
        "S_RT": "S_RT",
        "S_RT_CT": "S_RT_CT",
        "S": "S"  # This is a placeholder, adjust as needed
    }
    evaluate_natural_experiment_by_treatment(
        df,
        match_cols=["general.age", "tumor_characteristics.histological_diagnosis"],
        scenario_map=scenario_map
    )
    
    # 6) Influence-Function Risk Estimation
    print("\n--- Evaluating Influence-Function Risk by Treatment Scenario ---")
    for scenario in df['treatment_scenario'].unique():
        print(f"\n--- Scenario: {scenario} ---")
        evaluate_influence_function_risk(df, treatment_scenario=scenario)


if __name__ == "__main__":
    main()