from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class Outliercapping(BaseEstimator,TransformerMixin):
    def __init__(self,feature_list,lower_quantile=0.01,upper_quantile=0.99):
        self.feature_list = feature_list
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds = {}

    def fit(self,X,y=None):
        is_df=isinstance(X,pd.DataFrame)
        for i,col in enumerate(self.feature_list):
            data_col=X[col] if is_df else X[:,i]

            self.bounds[i] = (
                np.nanpercentile(data_col, self.lower_quantile * 100),
                np.nanpercentile(data_col, self.upper_quantile * 100))
        return self
    
    def transform(self,X):
        
        if isinstance(X,pd.DataFrame):
            X_array=X.values.copy()
            is_pandas=True
            original_columns=X.columns
            original_index=X.index
        else:
            X_array=np.array(X).copy()
            is_pandas=False

        for i in range(len((self.feature_list))):
            lower, upper = self.bounds[i]
            # Use np.clip to cap the values, if dataframe
            X_array[:,i]=np.clip(X_array[:,i],lower,upper)
        if is_pandas:
            return pd.DataFrame(X_array,columns=original_columns,index=original_index)
        return X_array


class year_handling(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X=X.copy()
        X["House_age"]=X["YrSold"]-X["YearBuilt"]
        X["Year_since_remodel"]=X["YrSold"]-X["YearRemodAdd"]
        X["Garage_age"] = X["YrSold"] - X["GarageYrBlt"]
        X['IsNew']      = (X['YrSold'] == X['YearBuilt']).astype(int)
        X['HasRemodel'] = (X['YearBuilt'] != X['YearRemodAdd']).astype(int)
        X['House_age'] = np.maximum(0, X['House_age'])
        X['Year_since_remodel'] = np.maximum(0, X['Year_since_remodel'])
        
        cols_to_drop=["YrSold","YearBuilt","YearRemodAdd","GarageYrBlt",'Id',"BsmtExposure"]

        X=X.drop(columns=[i for i in cols_to_drop if i in X.columns])
        return X

class area_features_handling(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X=X.copy()

        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        X['TotalArea'] = X['TotalSF'] + X['GarageArea'] + X['WoodDeckSF'] + X['OpenPorchSF'] + X['PoolArea']
        X['TotalBath'] = X['FullBath'] + 0.5*X['HalfBath'] + X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']

        X['TotalPorchSF'] = X[['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']].sum(axis=1)

        return X
class quality_feature_handling(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X=X.copy()

        X['Qual_x_TotalSF']   = X['OverallQual'] * X['TotalSF']
        X['Qual_x_GrLivArea'] = X['OverallQual'] * X['GrLivArea']

        return X
class NeighborhoodLowerCardinality(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=10):
        self.top_k = top_k
        
    def fit(self, X, y=None):
        self.top_categories_ = (
            X["Neighborhood"].value_counts()
            .nlargest(self.top_k)
            .index
        )
        return self
    
    def transform(self, X):
        X = X.copy()
        X['Neighborhood'] = X['Neighborhood'].apply(
            lambda x: x if x in self.top_categories_ else "other"
        )
        return X

# class NeighborhoodTargetEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, smoothing=5):
#         self.smoothing = smoothing
#         self.mapping_ = None
#         self.global_mean_ = None
        
#     def fit(self, X, y):
#         """
#         X: pd.DataFrame with column 'Neighborhood'
#         y: target array (log prices if using log scale)
#         """
#         X = X.copy()
#         y = pd.Series(y, name="target") 
#         self.global_mean_ = y.mean()
        
#         # Compute mean and counts per neighborhood
#         stats = (pd.concat([X[['Neighborhood']], y], axis=1).groupby('Neighborhood')['target'].agg(['mean', 'count']))
#         # Apply smoothing
#         stats['encoded'] = (stats['mean'] * stats['count'] + self.global_mean_ * self.smoothing) / (stats['count'] + self.smoothing)
#         self.mapping_ = stats['encoded'].to_dict()
#         return self
    
    # def transform(self, X):
    #     X = X.copy()
    #     # Map neighborhood to encoded value, unseen categories get global mean
    #     X['Neighborhood'] = X['Neighborhood'].map(self.mapping_).fillna(self.global_mean_)
    #     return X


