import pandas as pd
from pathlib import Path


class DataLoader:

    def __init__(self, file_path: str):

        self.file_path = file_path

        extension = Path(file_path).suffix.lower()

        if extension == ".csv":
            self.df = pd.read_csv(file_path)

        elif extension in [".xlsx", ".xls"]:
            self.df = pd.read_excel(file_path)

        elif extension == ".json":
            self.df = pd.read_json(file_path)

        elif extension == ".parquet":
            self.df = pd.read_parquet(file_path)

        else:
            raise ValueError(f"Unsupported file type : {extension}")

    def get_dataframe(self):
        return self.df