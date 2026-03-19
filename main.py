from data_loader import load_data
from preprocessing import preprocess_data
from model_training import train_models
from evaluation import evaluate_models
from prediction import predict_transaction
from visualization import plot_class_distribution

def main():
    # Step 1: Load data
    file_path = "data/dataset.csv"  # change path if needed
    data = load_data(file_path)

    if data is None:
        return

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Step 3: Train models
    models = train_models(X_train, y_train)

    # Step 4: Evaluate models
    evaluate_models(models, X_test, y_test)
