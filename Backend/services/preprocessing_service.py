from preprocessing.preprocesser import preprocessingEngine
from storage.dataset_store import dataset_store
import numpy as np
import pandas as pd

class PreprocessingService:

    def make_json_safe(self, obj):

        if isinstance(obj, dict):
            return {k: self.make_json_safe(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self.make_json_safe(v) for v in obj]

        elif isinstance(obj, tuple):
            return tuple(self.make_json_safe(v) for v in obj)

        elif isinstance(obj, pd.DataFrame):
            return self.make_json_safe(obj.to_dict(orient="index"))

        elif isinstance(obj, pd.Series):
            return self.make_json_safe(obj.to_dict())

        elif isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.dtype):
            return str(obj)

        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        elif pd.isna(obj):
            return None

        return obj


    def preprocess(self, dataset_id: str):

        if dataset_id not in dataset_store:
            raise ValueError("Dataset not found.")

        dataset = dataset_store[dataset_id]

        df = dataset["df"]
        analysis = dataset["analysis"]
        recommendation = dataset["recommendation"]

        if analysis is None:
            raise ValueError("Please run analysis first.")

        if recommendation is None:
            raise ValueError("Please generate recommendations first.")

        engine = preprocessingEngine(
            analysis,
            recommendation,
            df
        )

        preprocessing_summary, preprocessed_df = engine.proprocessor()

        dataset["preprocessed_data"] = preprocessed_df

        dataset["preprocessed_info"] = preprocessing_summary

        return {
            "preprocessing": self.make_json_safe(preprocessing_summary)
        }