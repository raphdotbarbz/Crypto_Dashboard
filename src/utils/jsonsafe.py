# src/utils/jsonsafe.py
import math
from typing import Any, Iterable
import numpy as np
import pandas as pd

def json_safe(obj: Any) -> Any:
    """Recursively convert NaN/Inf -> None and pandas/numpy types -> builtins."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, Iterable)) and not isinstance(obj, (str, bytes)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, pd.Series):
        return json_safe(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        df = obj.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notna(df), None)
        return json_safe(df.to_dict(orient="records"))
    return obj
