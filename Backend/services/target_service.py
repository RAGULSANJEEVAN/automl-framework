from dataset.target import TargetSelector
from storage.dataset_store import dataset_store


class TargetService:

    def select_target(self, dataset_id, target):

        if dataset_id not in dataset_store:
            raise ValueError("Dataset not found.")

        df = dataset_store[dataset_id]["df"]

        selector = TargetSelector(df)

        selected_target = selector.select_target(target)

        dataset_store[dataset_id]["target"] = selected_target

        return {
            "dataset_id": dataset_id,
            "target": selected_target
        }