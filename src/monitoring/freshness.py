from datetime import datetime
import pandas as pd

def check_data_freshness(latest_timestamp: pd.Timestamp | str, max_age_minutes: int = 60) -> tuple[bool, float]:
    """
    Returns (is_fresh, age_in_minutes) indicating if data is fresh enough for reliable predictions.
    """
    latest_ts = pd.to_datetime(latest_timestamp)
    now = pd.Timestamp.now(tz=latest_ts.tzinfo)  # Match timezone if present
    
    age_minutes = (now - latest_ts).total_seconds() / 60.0
    
    is_fresh = age_minutes <= max_age_minutes
    return is_fresh, age_minutes
