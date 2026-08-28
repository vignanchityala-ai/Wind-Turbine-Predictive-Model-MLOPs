"""Wraps training runs in MLflow context."""
import mlflow
import os

class ExperimentTracker:
    def __init__(self, experiment_name, tracking_uri=None):
        if tracking_uri is None:
            # Default to local mlruns if not provided
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def log_training_run(self, params, metrics, artifacts=None, model_name=None):
        with mlflow.start_run():
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)  # coverage, reliability, earliness, composite
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path)
            
            if model_name:
                # Optionally register model if model_name is provided
                # In a real setup, you'd log the model first. For now we just log metadata.
                pass

def check_quality_gate(new_metrics, production_metrics):
    """New model must beat production on composite score."""
    if not production_metrics:
        return True
    return new_metrics.get("composite", 0) >= production_metrics.get("composite", 0)
