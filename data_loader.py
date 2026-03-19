import pandas as pd

def load_data(C:\Users\addep\Downloads\archive.zip):
    try:
        data = pd.read_csv(C:\Users\addep\Downloads\archive.zip)
        print("Dataset loaded successfully!")
        print("Shape:", data.shape)
        return data
    except Exception as e:
        print("Error loading dataset:", e)
        return None
