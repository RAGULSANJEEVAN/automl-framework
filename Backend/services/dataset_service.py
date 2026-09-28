import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from storage.dataset_store import dataset_store


class DatasetService:

    def _serialize(self, obj):

        # ----------------------------
        # None
        # ----------------------------
        if obj is None:
            return None

        # ----------------------------
        # Python primitive types
        # ----------------------------
        if isinstance(obj, (str, int, float, bool)):
            return obj

        # ----------------------------
        # NumPy types
        # ----------------------------
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.bool_):
            return bool(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # ----------------------------
        # Pandas Timestamp
        # ----------------------------
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        # ----------------------------
        # Pandas Series
        # ----------------------------
        if isinstance(obj, pd.Series):
            return self._serialize(obj.to_dict())

        # ----------------------------
        # Pandas Index
        # ----------------------------
        if isinstance(obj, pd.Index):
            return obj.tolist()

        # ----------------------------
        # Pandas DataFrame
        # ----------------------------
        if isinstance(obj, pd.DataFrame):

            return {
                "rows": int(obj.shape[0]),
                "columns": int(obj.shape[1]),
                "column_names": obj.columns.tolist(),
                "preview": obj.head(10).replace({np.nan: None}).to_dict(orient="records")
            }

        # ----------------------------
        # sklearn model
        # ----------------------------
        if isinstance(obj, BaseEstimator):
            return obj.__class__.__name__

        # ----------------------------
        # Dictionary
        # ----------------------------
        if isinstance(obj, dict):

            return {
                str(k): self._serialize(v)
                for k, v in obj.items()
            }

        # ----------------------------
        # List / Tuple / Set
        # ----------------------------
        if isinstance(obj, (list, tuple, set)):

            return [
                self._serialize(v)
                for v in obj
            ]

        # ----------------------------
        # Objects with __dict__
        # ----------------------------
        if hasattr(obj, "__dict__"):

            return {
                "__class__": obj.__class__.__name__,
                **{
                    k: self._serialize(v)
                    for k, v in vars(obj).items()
                }
            }

        # ----------------------------
        # Fallback
        # ----------------------------
        return str(obj)

    def get_dataset(self, dataset_id):

        if dataset_id not in dataset_store:
            raise ValueError("Dataset not found.")

        return self._serialize(dataset_store[dataset_id])