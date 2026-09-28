from dataset.loader import DataLoader
from dataset.target import TargetSelector
from dataset.analyzer import DatasetAnalyzer
from recommendation.recommender import RecommendationEngine
from preprocessing.preprocesser import preprocessingEngine
from modelTraining.model_training import modelTrainingEngine

loader = DataLoader()

# file_path = input("Enter Dataset Path : ")


df = loader.get_dataframe()

column_details = []

for col in df.columns:

    column_details.append({
        "name": col,
        "type": str(df[col].dtype)
    })

print(df.head())

selector = TargetSelector(df)

target = selector.select_target()

print("Target :", target)

analyzer = DatasetAnalyzer(df,target)

analysis = analyzer.analyze()

print(analysis)

recommender = RecommendationEngine(analysis, df)

recommender =  recommender.recommend()

print("recommender", recommender)

preprocessor = preprocessingEngine(analysis, recommender, df )

preprocessing, preprocessed_data  = preprocessor.proprocessor()

print("preprocessing",preprocessed_data)

problem_type = analysis["problem_info"]["problem_type"]

if problem_type == "supervised_learning":

    data = preprocessor.split_data()

else:

    data = preprocessor.df_preprocessing

modelTraining = modelTrainingEngine(analysis, recommender, data)

model_training = modelTraining.modelTraining()

print("model Training",model_training )