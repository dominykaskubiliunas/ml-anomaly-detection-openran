import pandas as pd

class DataPrep():
    def __init__(self):               
        pass

    def convert_to_seconds_sample(self, timestamp):
        
        minutes, seconds = map(float, timestamp.split(':'))
        return minutes * 60 + seconds

    def normalize_time_and_convert_to_seconds(self, df, column_name):

        df[column_name] = pd.to_datetime(df[column_name])
        df[column_name] = df[column_name].astype('int64') / 1e9
        smallest_time = df[column_name].min()
        df[column_name] = df[column_name].apply(lambda x: x - smallest_time)

        return df

    def normalize_time_and_convert_to_seconds_sample(self, df, column_name):
        
        df[column_name] = df[column_name].apply(self.convert_to_seconds_sample)
        smallest_time = df[column_name].min()
        df[column_name] = df[column_name].apply(lambda x: x - smallest_time)

        return df
    def split_data_by_time(self, start, end, data):

        return data[(data['timestamp'] >= start) & (data['timestamp'] <= end)]   
        

    def fill_empty_rows(self, df):

        return df.ffill()