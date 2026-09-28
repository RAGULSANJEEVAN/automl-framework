from modelTraining.model_training import modelTrainingEngine
from storage.dataset_store import dataset_store
import numpy as np
import pandas as pd

class TrainingService:

    def debug_json(obj, path="root"):

        if isinstance(obj, dict):
            for k, v in obj.items():
                debug_json(v, f"{path}.{k}")

        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                debug_json(v, f"{path}[{i}]")

        else:

            if isinstance(obj, pd.DataFrame):
                print(path, "-> DataFrame")

            elif isinstance(obj, pd.Series):
                print(path, "-> Series")

            elif isinstance(obj, pd.Index):
                print(path, "-> Index")

            elif isinstance(obj, np.ndarray):
                print(path, "-> ndarray")

            elif isinstance(obj, np.integer):
                print(path, "-> np.integer")

            elif isinstance(obj, np.floating):
                print(path, "-> np.float")

            elif hasattr(obj, "get_params"):
                print(path, "-> sklearn:", obj.__class__.__name__)
    
    def train(self, dataset_id: str):

        if dataset_id not in dataset_store:
            raise ValueError("Dataset not found.")

        dataset = dataset_store[dataset_id]

        analysis = dataset["analysis"]
        recommendation = dataset["recommendation"]
        data = dataset["preprocessed_data"]

        if analysis is None:
            raise ValueError("Please run analysis first.")

        if recommendation is None:
            raise ValueError("Please run recommendation first.")

        if data is None:
            raise ValueError("Please run preprocessing first.")

        trainer = modelTrainingEngine(
            analysis,
            recommendation,
            data
        )

        training_result = trainer.modelTraining()

        dataset["training_result"] = training_result

        print(f"Training for dataset : {dataset}")

        # return self.debug_json(training_result)

        return training_result


