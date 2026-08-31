"""Feature drift detection using PSI (Population Stability Index)."""
import numpy as np

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between two distributions.
    PSI > 0.2 typically indicates significant drift.
    """
    # Create unified bin edges spanning both distributions
    min_val = min(np.min(reference), np.min(current))
    max_val = max(np.max(reference), np.max(current))
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Calculate histograms
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)
    
    # Convert to proportions
    ref_props = ref_counts / max(len(reference), 1)
    curr_props = curr_counts / max(len(current), 1)
    
    # Add small epsilon to avoid log(0) and division by zero
    eps = 1e-4
    ref_props = np.maximum(ref_props, eps)
    curr_props = np.maximum(curr_props, eps)
    
    # Compute PSI
    psi_values = (curr_props - ref_props) * np.log(curr_props / ref_props)
    return float(np.sum(psi_values))

def check_drift(reference_features: dict, current_features: dict, threshold: float = 0.2) -> list[str]:
    """
    Check all features for drift. 
    Returns list of drifted feature names where PSI > threshold.
    """
    drifted_features = []
    
    for feature_name, ref_data in reference_features.items():
        if feature_name in current_features:
            curr_data = current_features[feature_name]
            psi = compute_psi(np.array(ref_data), np.array(curr_data))
            if psi > threshold:
                drifted_features.append(feature_name)
                
    return drifted_features

def should_retrain(drift_results: dict[str, bool], model_age_days: int, max_age: int = 30) -> bool:
    """
    Trigger retraining if drift detected (any feature drifted) or model is too old.
    drift_results maps feature names to a boolean (True if drifted).
    """
    is_drifted = any(drift_results.values())
    is_too_old = model_age_days > max_age
    
    return is_drifted or is_too_old
