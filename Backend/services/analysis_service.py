from dataset.analyzer import DatasetAnalyzer
from storage.dataset_store import dataset_store


class AnalysisService:

    def analyze_dataset(self, dataset_id: str):

        if dataset_id not in dataset_store:
            raise ValueError("Dataset not found.")

        dataset = dataset_store[dataset_id]

        df = dataset["df"]
        target = dataset["target"]

        analyzer = DatasetAnalyzer(df, target)

        analysis = analyzer.analyze()

        dataset["analysis"] = analysis

        return analysis