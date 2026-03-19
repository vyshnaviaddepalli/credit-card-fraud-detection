import pandas as pd
file_path = "creditcard.csv"
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        print("Shape:", data.shape)
        return data
    except Exception as e:
        print("Error loading dataset:", e)
        return None
