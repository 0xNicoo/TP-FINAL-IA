import pandas as pd
from sklearn.model_selection import train_test_split

def read_csv(ruta):
    return pd.read_csv(ruta, delimiter=";")

def process_data(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].str.replace(',', '.')
    return data

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)