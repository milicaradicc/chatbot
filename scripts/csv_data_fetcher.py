import pandas as pd

class CSVDataFetcher:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)

    def fetch_and_process_csv(self):
        rows = []
        for _, row in self.df.iterrows():
            row_string = ', '.join([f"{col}: {row[col]}" for col in self.df.columns])
            rows.append(row_string)
        return rows
