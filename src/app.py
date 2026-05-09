import pandas as pd
import joblib
from src.data_loader import load_data

def predict():
    # Load the entire fitted pipeline (includes preprocessing!)
    model_pipeline = joblib.load('models/housing_pipeline.joblib')
    
    # Load raw test data
    _, test_raw = load_data(None, 'data/test.csv')
    
    # Predict directly on raw data
    # The pipeline automatically applies the imputer, scaler, and encoder
    predictions = model_pipeline.predict(test_raw)
    
    output = pd.DataFrame({'Id': test_raw['Id'], 'SalePrice': predictions})
    output.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    predict()