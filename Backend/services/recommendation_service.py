from recommendation.recommender import RecommendationEngine
from storage.dataset_store import dataset_store


class RecommendationService:

    def recommend(self, dataset_id: str):

        if dataset_id not in dataset_store:
            raise ValueError("Dataset not found.")

        dataset = dataset_store[dataset_id]

        df = dataset["df"]

        analysis = dataset["analysis"]

        if analysis is None:
            raise ValueError(
                "Dataset analysis not found. Please run analysis first."
            )

        recommender = RecommendationEngine(
            analysis,
            df
        )

        recommendation = recommender.recommend()

        dataset["recommendation"] = recommendation
        
        return recommendation