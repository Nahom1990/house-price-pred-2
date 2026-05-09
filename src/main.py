import joblib
from src.data_loader import load_data, get_features_target
from src.pipeline import get_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

def run_training():
    # 1. Load and Split
    train_raw, _ = load_data('data/train.csv')
    X, y = get_features_target(train_raw)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Build Pipeline
    base_model = GradientBoostingRegressor(random_state=42)
    pipeline = get_pipeline(base_model)

    # 3. Hyperparameter Tuning
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_root_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # 4. Save Final Pipeline
    joblib.dump(grid_search.best_estimator_, 'models/housing_pipeline.joblib')
    print("Model saved to models/housing_pipeline.joblib")

if __name__ == "__main__":
    run_training()