import pandas as pd

def load_data(train_path, test_path=None):
    """Loads the housing dataset."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    return train_df, test_df

def get_features_target(df, target_col='SalePrice'):
    """Separates features and target, dropping the Id column."""
    X = df.drop(columns=[target_col, 'Id'], errors='ignore')
    y = df[target_col] if target_col in df.columns else None
    return X, y