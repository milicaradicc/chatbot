import pandas as pd

class CSVDataFetcher:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        try:
            self.df = pd.read_csv(csv_file)
            date_columns = ['Start Date', 'End Date']
            for col in date_columns:
                self.df[col] = pd.to_datetime(self.df[col]).dt.strftime('%Y-%m-%d')
            time_columns = ['Start Time', 'End Time']
            for col in time_columns:
                self.df[col] = pd.to_datetime(self.df[col], format='%H:%M').dt.strftime('%H:%M')
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")


    def fetch_and_process_csv(self):
        rows = []
        for _, row in self.df.iterrows():
            row_string = ', '.join([f"{col}: {row[col]}" for col in self.df.columns])
            rows.append(row_string)
        return rows
