import uuid
import os

from dataset.loader import DataLoader
from storage.dataset_store import dataset_store

class UploadService:

    def __init__(self):

        self.upload_folder = "uploads"

        os.makedirs(self.upload_folder, exist_ok=True)

    def upload_dataset(self, file):

        extension = file.filename.split(".")[-1]

        dataset_id = str(uuid.uuid4())

        file_name = f"{file.filename}_{dataset_id}.{extension}"

        file_path = os.path.join(
            self.upload_folder,
            file_name
        )

        with open(file_path, "wb") as f:
            f.write(file.file.read())

        loader = DataLoader(file_path)

        df = loader.get_dataframe()

        dataset_store[dataset_id] = {
            "df": df,
            "file_path": file_path,
            "target": None,
            "analysis": None,
            "recommendation": None,
            "preprocessed_data": None,
            "training_result": None
        }

        column_details = []

        for col in df.columns: 

            column_details.append({
                "name": col,
                "type": str(df[col].dtype)
            })

        return {
            "dataset_id": dataset_id,
            "file_name": file.filename,
            "file_path": file_path,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns),
            "column_details": column_details
        }