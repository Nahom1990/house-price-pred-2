# House Price Prediction - Notebook Overview

## Project Summary
This project uses the `housing_price.ipynb` notebook to build a robust machine learning pipeline for predicting house prices. The workflow follows best practices for regression modeling, including data exploration, feature engineering, preprocessing, model selection, evaluation, and deployment.

## Workflow Outline

1. **Define Problem & Metric**
   - Identify the prediction target (e.g., SalePrice) and evaluation metric (e.g., RMSE, MAE, R²).

2. **Exploratory Data Analysis (EDA)**
   - Inspect data types, missing values, and distributions.
   - Visualize target and feature distributions, correlations, and missingness patterns.
   - Example visualizations: histograms, boxplots, scatterplots, heatmaps, missingno matrix.

3. **Create Hold-out Test Set**
   - Split data into training and test sets (e.g., 80/20) to ensure honest evaluation.

4. **Feature Engineering**
   - Create new features (e.g., ratios, date parts, interaction terms).
   - Avoid target leakage; implement feature engineering as transformers when possible.

5. **Preprocessing Pipelines**
   - Build `ColumnTransformer` pipelines for numeric and categorical features.
   - Numeric: imputation, scaling.
   - Categorical: imputation, encoding.
   - Combine all steps in a `Pipeline` for reproducibility and to prevent data leakage.

6. **Feature Selection & Dimensionality Reduction**
   - Apply feature selection (e.g., RFE, SelectKBest) and/or PCA inside the pipeline after preprocessing.

7. **Baseline Models & Cross-Validation**
   - Train simple models (mean regressor, linear regression, tree-based) using cross-validation.
   - Compare models using CV metrics.

8. **Model Selection & Hyperparameter Tuning**
   - Use `RandomizedSearchCV` or `GridSearchCV` for hyperparameter tuning on top models.
   - Always tune within the pipeline to avoid leakage.

9. **Final Evaluation**
   - Evaluate the best model on the hold-out test set.
   - Generate diagnostic plots (residuals, predicted vs actual).

10. **Interpretation & Diagnostics**
    - Analyze feature importance (permutation, SHAP), residuals, and error segments.

11. **Save & Deploy Model**
    - Save the entire pipeline with `joblib` for future predictions.

## Key Best Practices
- **All preprocessing and feature engineering should be inside the pipeline.**
- **Never fit any transformer or model on the test set.**
- **Use cross-validation for model comparison and tuning.**
- **Document each step and visualize results for transparency.**

## Example Code Skeleton
```python
# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define pipelines
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preproc = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# 3. Full pipeline
pipe = Pipeline([
    ("preproc", preproc),
    ("feat_sel", SelectFromModel(RandomForestRegressor(n_estimators=100))),
    ("pca", PCA(n_components=20)),  # optional
    ("model", RandomForestRegressor())
])

# 4. Cross-validation
scores = cross_validate(pipe, X_train, y_train, cv=5, scoring=("r2","neg_root_mean_squared_error"))

# 5. Hyperparameter search
param_dist = {...}
search = RandomizedSearchCV(pipe, param_distributions=param_dist, cv=5, n_iter=50)
search.fit(X_train, y_train)

# 6. Final evaluation
best = search.best_estimator_
y_pred = best.predict(X_test)
print(r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False))

# 7. Save pipeline
joblib.dump(best, "final_pipeline.joblib")
```

## Files in This Project
- `housing_price.ipynb`: Main notebook with the full workflow.
- `train.csv`, `test.csv`: Training and test datasets.
- `final_stacked_model.joblib`, `final_test_model.joblib`: Saved models.
- `custom_transformers.py`: Custom feature engineering or preprocessing transformers.
- `src/`: Source code for modular pipeline components.

## References
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [Pandas documentation](https://pandas.pydata.org/)
- [Seaborn documentation](https://seaborn.pydata.org/)

---
*This README summarizes the workflow and best practices implemented in `housing_price.ipynb` for reproducible, high-quality regression modeling.*
