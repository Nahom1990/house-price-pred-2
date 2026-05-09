from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA

def get_pipeline(model, n_pca=None):
    # Lists derived from your feature engineering steps
    num_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'GrLivArea', 'GarageArea'] 
    cat_cols = ['MSZoning', 'Street', 'Neighborhood', 'HouseStyle']

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    steps = [('preprocessor', preprocessor)]
    
    if n_pca:
        steps.append(('pca', PCA(n_components=n_pca)))
    
    steps.append(('model', model))
    
    return Pipeline(steps)