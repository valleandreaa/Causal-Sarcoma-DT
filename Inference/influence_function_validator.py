"""
Influence Function (IF) Two-Step Validator for CycleGAN Treatment Effect Validation

This module implements the influence function-based validation approach for
estimating Precision in Estimation of Heterogeneous Effects (PEHE) without
requiring counterfactual outcomes. The method uses a two-step process:

1. Step 1 - Plug-in loss: Uses only factual data to build nuisance models
2. Step 2 - IF correction: Adds an influence function term to remove bias

References:
- Alaa et al. "Validating Causal Inference Models via Influence Functions" ICML 2019
- Chernozhukov et al. "Double/debiased machine learning for treatment and structural parameters"

Key Components:
- Multi-arm treatment support (surgery, chemotherapy, radiotherapy)
- Cross-fitting to maintain sample independence
- Efficient influence function computation
- PEHE estimation without counterfactual labels
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings


class NuisanceModels:
    """Container for nuisance models used in IF validation."""
    
    def __init__(self, n_treatments: int = 3):
        """
        Initialize nuisance models.
        
        Parameters
        ----------
        n_treatments : int
            Number of treatment arms (default: 3 for surgery, chemo, radio)
        """
        self.n_treatments = n_treatments
        self.outcome_models = {}  # mu_hat: outcome models per treatment arm
        self.propensity_model = None  # pi_hat: propensity score model
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> 'NuisanceModels':
        """
        Fit nuisance models on training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix (clinical embeddings)
        W : np.ndarray, shape (n_samples,)
            Treatment assignments (0, 1, 2, ...)
        Y : np.ndarray, shape (n_samples,)
            Outcomes
            
        Returns
        -------
        self : NuisanceModels
            Fitted instance
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit outcome models for each treatment arm
        for w in range(self.n_treatments):
            mask = (W == w)
            if np.sum(mask) > 0:  # Ensure we have samples for this treatment
                self.outcome_models[w] = RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42,
                    min_samples_leaf=5
                ).fit(X_scaled[mask], Y[mask])
            else:
                # Fallback: fit on all data but warn
                warnings.warn(f"No samples for treatment {w}, using all data for outcome model")
                self.outcome_models[w] = RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42,
                    min_samples_leaf=5
                ).fit(X_scaled, Y)
        
        # Fit propensity score model
        self.propensity_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=5
        ).fit(X_scaled, W)
        
        self.is_fitted = True
        return self
    
    def predict_outcomes(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Predict outcomes for each treatment arm.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
            
        Returns
        -------
        Dict[int, np.ndarray]
            Predicted outcomes for each treatment arm
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for w in range(self.n_treatments):
            if w in self.outcome_models:
                predictions[w] = self.outcome_models[w].predict(X_scaled)
            else:
                # Fallback: return zeros
                predictions[w] = np.zeros(X.shape[0])
        
        return predictions
    
    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        """
        Predict propensity scores for all treatment arms.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
            
        Returns
        -------
        np.ndarray, shape (n_samples, n_treatments)
            Propensity scores for each treatment arm
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.propensity_model.predict_proba(X_scaled)


def compute_plug_in_cate(outcome_predictions: Dict[int, np.ndarray], 
                        reference_arm: int = 0) -> np.ndarray:
    """
    Compute plug-in Conditional Average Treatment Effect (CATE).
    
    Parameters
    ----------
    outcome_predictions : Dict[int, np.ndarray]
        Predicted outcomes for each treatment arm
    reference_arm : int
        Reference treatment arm (default: 0)
        
    Returns
    -------
    np.ndarray
        Plug-in CATE estimates (treatment - control)
    """
    n_samples = len(next(iter(outcome_predictions.values())))
    cate_estimates = np.zeros(n_samples)
    
    # Compute CATE as difference from reference arm
    ref_outcomes = outcome_predictions.get(reference_arm, np.zeros(n_samples))
    
    for w, outcomes in outcome_predictions.items():
        if w != reference_arm:
            cate_estimates += (outcomes - ref_outcomes)
    
    # Average across non-reference arms
    n_treatment_arms = len([w for w in outcome_predictions.keys() if w != reference_arm])
    if n_treatment_arms > 0:
        cate_estimates /= n_treatment_arms
    
    return cate_estimates


def compute_influence_function(X: np.ndarray, W: np.ndarray, Y: np.ndarray,
                             T_hat: np.ndarray, T_tilde: np.ndarray,
                             propensity_scores: np.ndarray,
                             n_treatments: int = 3) -> np.ndarray:
    """
    Compute the efficient influence function for PEHE estimation.
    
    This implements the multi-arm version of the influence function from
    Alaa et al. (2019) for K treatments.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix
    W : np.ndarray, shape (n_samples,)
        Treatment assignments
    Y : np.ndarray, shape (n_samples,)
        Outcomes
    T_hat : np.ndarray, shape (n_samples,)
        CycleGAN treatment effect predictions
    T_tilde : np.ndarray, shape (n_samples,)
        Plug-in treatment effect estimates
    propensity_scores : np.ndarray, shape (n_samples, n_treatments)
        Propensity scores for each treatment
    n_treatments : int
        Number of treatment arms
        
    Returns
    -------
    np.ndarray, shape (n_samples,)
        Influence function values
    """
    n_samples = len(X)
    influence_values = np.zeros(n_samples)
    
    # Compute plug-in PEHE for bias correction
    pehe_plugin = np.mean((T_hat - T_tilde) ** 2)
    
    for i in range(n_samples):
        w_i = int(W[i])
        y_i = Y[i]
        t_hat_i = T_hat[i]
        t_tilde_i = T_tilde[i]
        pi_i = propensity_scores[i]  # shape (n_treatments,)
        
        # Ensure propensity scores are bounded away from 0 and 1
        pi_i = np.clip(pi_i, 1e-6, 1 - 1e-6)
        
        influence_i = 0.0
        
        for k in range(n_treatments):
            # Compute A_k and B_k terms
            A_k = (1.0 if w_i == k else 0.0) - pi_i[k]
            
            # Avoid division by zero
            if pi_i[k] > 1e-6 and pi_i[k] < 1 - 1e-6:
                B_k = 2 * (1.0 if w_i == k else 0.0) * A_k / (pi_i[k] * (1 - pi_i[k]))
            else:
                B_k = 0.0
            
            # Compute influence function components
            diff = t_hat_i - t_tilde_i
            term1 = (1 - B_k) * (t_tilde_i ** 2)
            term2 = -A_k * (diff ** 2)
            term3 = B_k * y_i * diff
            
            influence_i += term1 + term2 + term3
        
        # Subtract plug-in PEHE (bias correction)
        influence_values[i] = influence_i - pehe_plugin
    
    return influence_values


def if_validated_pehe(X: np.ndarray, W: np.ndarray, Y: np.ndarray,
                     T_hat: np.ndarray, nuisance_models: NuisanceModels,
                     n_treatments: int = 3) -> Tuple[float, Dict[str, float]]:
    """
    Compute influence function validated PEHE.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    W : np.ndarray
        Treatment assignments
    Y : np.ndarray
        Outcomes
    T_hat : np.ndarray
        CycleGAN treatment effect predictions
    nuisance_models : NuisanceModels
        Fitted nuisance models
    n_treatments : int
        Number of treatment arms
        
    Returns
    -------
    Tuple[float, Dict[str, float]]
        IF-validated PEHE and diagnostic metrics
    """
    # Step 1: Compute plug-in CATE
    outcome_predictions = nuisance_models.predict_outcomes(X)
    T_tilde = compute_plug_in_cate(outcome_predictions)
    
    # Compute plug-in PEHE
    pehe_plugin = np.mean((T_hat - T_tilde) ** 2)
    
    # Step 2: Compute influence function correction
    propensity_scores = nuisance_models.predict_propensity(X)
    influence_values = compute_influence_function(
        X, W, Y, T_hat, T_tilde, propensity_scores, n_treatments
    )
    
    # IF correction term
    if_correction = np.mean(influence_values)
    
    # Final IF-validated PEHE
    pehe_if = pehe_plugin + if_correction
    
    # Diagnostic metrics
    diagnostics = {
        "pehe_plugin": float(pehe_plugin),
        "if_correction": float(if_correction),
        "if_correction_magnitude": float(np.abs(if_correction)),
        "correction_ratio": float(np.abs(if_correction) / (pehe_plugin + 1e-8)),
        "influence_std": float(np.std(influence_values)),
        "propensity_min": float(np.min(propensity_scores)),
        "propensity_max": float(np.max(propensity_scores)),
        "overlap_violation_rate": float(np.mean(
            (np.min(propensity_scores, axis=1) < 0.05) | 
            (np.max(propensity_scores, axis=1) > 0.95)
        ))
    }
    
    return float(pehe_if), diagnostics


class CrossFittedIFValidator:
    """
    Cross-fitted influence function validator for CycleGAN treatment effects.
    
    This class implements the full cross-fitting procedure to ensure
    sample independence between estimation and evaluation.
    """
    
    def __init__(self, n_splits: int = 5, n_treatments: int = 3, 
                 random_state: int = 42):
        """
        Initialize cross-fitted validator.
        
        Parameters
        ----------
        n_splits : int
            Number of cross-validation folds
        n_treatments : int
            Number of treatment arms
        random_state : int
            Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.n_treatments = n_treatments
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def validate(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray,
                T_hat: np.ndarray) -> Tuple[float, List[float], Dict[str, Any]]:
        """
        Perform cross-fitted IF validation.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Features (clinical embeddings)
        W : np.ndarray, shape (n_samples,)
            Treatment assignments
        Y : np.ndarray, shape (n_samples,)
            Outcomes
        T_hat : np.ndarray, shape (n_samples,)
            CycleGAN treatment effect predictions
            
        Returns
        -------
        Tuple[float, List[float], Dict[str, Any]]
            Mean IF-validated PEHE, fold-wise PEHEs, and aggregated diagnostics
        """
        fold_pehes = []
        fold_diagnostics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.kf.split(X)):
            print(f"Processing fold {fold_idx + 1}/{self.n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            W_train, W_val = W[train_idx], W[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            T_hat_val = T_hat[val_idx]
            
            # Fit nuisance models on training fold
            nuisance_models = NuisanceModels(n_treatments=self.n_treatments)
            nuisance_models.fit(X_train, W_train, Y_train)
            
            # Compute IF-validated PEHE on validation fold
            pehe_fold, diagnostics_fold = if_validated_pehe(
                X_val, W_val, Y_val, T_hat_val, nuisance_models, self.n_treatments
            )
            
            fold_pehes.append(pehe_fold)
            fold_diagnostics.append(diagnostics_fold)
            
            print(f"  Fold {fold_idx + 1} PEHE: {pehe_fold:.6f}")
            print(f"  Plug-in: {diagnostics_fold['pehe_plugin']:.6f}, "
                  f"IF correction: {diagnostics_fold['if_correction']:.6f}")
        
        # Aggregate results
        mean_pehe = np.mean(fold_pehes)
        
        # Aggregate diagnostics
        aggregated_diagnostics = {
            "fold_pehes": fold_pehes,
            "pehe_std": float(np.std(fold_pehes)),
            "mean_plugin_pehe": float(np.mean([d["pehe_plugin"] for d in fold_diagnostics])),
            "mean_if_correction": float(np.mean([d["if_correction"] for d in fold_diagnostics])),
            "mean_correction_magnitude": float(np.mean([d["if_correction_magnitude"] for d in fold_diagnostics])),
            "mean_correction_ratio": float(np.mean([d["correction_ratio"] for d in fold_diagnostics])),
            "mean_overlap_violations": float(np.mean([d["overlap_violation_rate"] for d in fold_diagnostics])),
            "n_folds": self.n_splits,
            "n_samples": len(X),
            "n_treatments": self.n_treatments
        }
        
        return mean_pehe, fold_pehes, aggregated_diagnostics


def extract_features_from_loader(loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features, treatments, and outcomes from a DataLoader.
    
    Parameters
    ----------
    loader : DataLoader
        PyTorch DataLoader containing patient data
    device : torch.device
        Computing device
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Features (clinical embeddings), treatments, outcomes
    """
    features_list = []
    treatments_list = []
    outcomes_list = []
    
    with torch.no_grad():
        for treat, clin, actual in loader:
            # Average clinical embeddings across time steps
            clin_mean = clin.mean(dim=1).cpu().numpy()  # (batch_size, embedding_dim)
            features_list.append(clin_mean)
            
            # Extract treatment information (take last timestep or mode)
            actual_np = actual.cpu().numpy()
            if len(actual_np.shape) > 1:
                # Take last timestep or most common treatment
                treatments = actual_np[:, -1] if actual_np.shape[1] > 0 else actual_np[:, 0]
            else:
                treatments = actual_np
            treatments_list.append(treatments)
            
            # Mock outcomes (in practice, extract from your survival data)
            # This should be replaced with actual outcome extraction
            outcomes = np.random.normal(0, 1, size=len(treatments))  # Placeholder
            outcomes_list.append(outcomes)
    
    X = np.vstack(features_list) if features_list else np.array([])
    W = np.concatenate(treatments_list) if treatments_list else np.array([])
    Y = np.concatenate(outcomes_list) if outcomes_list else np.array([])
    
    return X, W, Y


def validate_cyclegan_with_if(loader: DataLoader, gx, decoder_treat,
                             feat_spec: Dict[str, Any], oh_map: Dict[str, Any],
                             ord_map: Dict[str, Any], num_ranges: Dict[str, Any],
                             device: torch.device, outcome_extractor=None) -> Dict[str, Any]:
    """
    Main function to validate CycleGAN using influence function approach.
    
    Parameters
    ----------
    loader : DataLoader
        Patient data loader
    gx : torch.nn.Module
        CycleGAN generator
    decoder_treat : torch.nn.Module
        Treatment decoder
    feat_spec, oh_map, ord_map, num_ranges : Dict
        Feature specifications
    device : torch.device
        Computing device
    outcome_extractor : callable, optional
        Function to extract outcomes from patient data
        
    Returns
    -------
    Dict[str, Any]
        IF validation results and diagnostics
    """
    from Models.models.utils import randomize_one_hot
    from Models.models.decoder_embedders import decode_embedding
    
    print("="*60)
    print("INFLUENCE FUNCTION VALIDATION")
    print("="*60)
    
    # Extract features and outcomes
    print("Extracting features and outcomes...")
    X, W, Y = extract_features_from_loader(loader, device)
    
    if len(X) == 0:
        return {"error": "No data extracted from loader"}
    
    print(f"Extracted {len(X)} samples with {X.shape[1]} features")
    
    # Generate CycleGAN treatment effect predictions
    print("Generating CycleGAN treatment effect predictions...")
    T_hat_list = []
    
    gx.eval()
    with torch.no_grad():
        for treat, clin, actual in loader:
            treat = treat.to(device).float()
            clin = clin.to(device).float()
            actual = actual.to(device)
            
            # Generate counterfactual treatments for different scenarios
            batch_effects = []
            n_scenarios = 3  # Number of counterfactual scenarios
            
            for scenario in range(n_scenarios):
                tr_counter = randomize_one_hot(actual)
                fake_embeddings = gx(clin, tr_counter)
                
                # Decode and compute treatment effect proxy
                # (In practice, this should be a meaningful treatment effect metric)
                effect_proxy = fake_embeddings.mean().item()  # Simplified proxy
                batch_effects.append(effect_proxy)
            
            # Use variance across scenarios as treatment effect estimate
            T_hat_batch = np.var(batch_effects) if len(batch_effects) > 1 else 0.0
            T_hat_list.extend([T_hat_batch] * len(treat))
    
    T_hat = np.array(T_hat_list)
    
    # Run cross-fitted IF validation
    print("Running cross-fitted influence function validation...")
    validator = CrossFittedIFValidator(n_splits=3, n_treatments=3)  # Reduced splits for smaller datasets
    
    try:
        pehe_if, fold_pehes, diagnostics = validator.validate(X, W, Y, T_hat)
        
        print(f"IF-validated PEHE: {pehe_if:.6f} ± {diagnostics['pehe_std']:.6f}")
        print(f"Mean correction magnitude: {diagnostics['mean_correction_magnitude']:.6f}")
        print(f"Overlap violation rate: {diagnostics['mean_overlap_violations']:.3f}")
        
        return {
            "if_validated_pehe": pehe_if,
            "pehe_std": diagnostics['pehe_std'],
            "fold_pehes": fold_pehes,
            "diagnostics": diagnostics,
            "validation_successful": True
        }
        
    except Exception as e:
        print(f"IF validation failed: {e}")
        return {
            "error": str(e),
            "validation_successful": False
        }


def print_if_validation_summary(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of IF validation results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from validate_cyclegan_with_if
    """
    print("\n" + "="*80)
    print("INFLUENCE FUNCTION VALIDATION SUMMARY")
    print("="*80)
    
    if not results.get("validation_successful", False):
        print(f"❌ VALIDATION FAILED: {results.get('error', 'Unknown error')}")
        return
    
    pehe = results["if_validated_pehe"]
    pehe_std = results["pehe_std"]
    diagnostics = results["diagnostics"]
    
    print(f"\n🎯 IF-VALIDATED PEHE: {pehe:.6f} ± {pehe_std:.6f}")
    
    # Interpretation
    print(f"\n📊 DIAGNOSTIC METRICS:")
    print(f"   • Mean plug-in PEHE: {diagnostics['mean_plugin_pehe']:.6f}")
    print(f"   • Mean IF correction: {diagnostics['mean_if_correction']:.6f}")
    print(f"   • Correction magnitude: {diagnostics['mean_correction_magnitude']:.6f}")
    print(f"   • Correction ratio: {diagnostics['mean_correction_ratio']:.3f}")
    print(f"   • Overlap violations: {diagnostics['mean_overlap_violations']:.1%}")
    
    # Quality assessment
    correction_ratio = diagnostics['mean_correction_ratio']
    overlap_violations = diagnostics['mean_overlap_violations']
    
    print(f"\n🔍 QUALITY ASSESSMENT:")
    
    if correction_ratio < 0.1:
        print("   ✅ Small IF correction - nuisance models are reliable")
    elif correction_ratio < 0.5:
        print("   ⚠️  Moderate IF correction - consider improving nuisance models")
    else:
        print("   ❌ Large IF correction - nuisance models may be poor")
    
    if overlap_violations < 0.05:
        print("   ✅ Good overlap - positivity assumption satisfied")
    elif overlap_violations < 0.20:
        print("   ⚠️  Moderate overlap violations - some extrapolation")
    else:
        print("   ❌ High overlap violations - strong extrapolation required")
    
    print(f"\n📈 FOLD-WISE RESULTS:")
    for i, fold_pehe in enumerate(results["fold_pehes"]):
        print(f"   Fold {i+1}: {fold_pehe:.6f}")
    
    print("\n" + "="*80)
