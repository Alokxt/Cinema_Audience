from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from src.feature_extraction.final_features_pipeline import FeatureGenerator , get_feature_data

booknow_theaters,weekday_tcktsbook, weekday_tcktsold,month_tcktsold, month_tcktsbooked,mean_audiences = get_feature_data()

f2 = FeatureGenerator(booknow_theaters,weekday_tcktsbook, weekday_tcktsold,month_tcktsold, month_tcktsbooked,mean_audiences)

num_features = ['mean_audience','prior_info','day','tkt_book_month',
       'tkt_sold_month', 'tkt_book_week', 'tkt_sold_week',
       'is_summer', 'is_weekend','is_dec','longitude','latitude','lag1','lag7']
cat_features = ['week_day','year','theater_type']

num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='mean')),
    
])

cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('numeric',num_pipe,num_features),
        ('categorical',cat_pipe,cat_features)
    ]
)

full_pipe = Pipeline([
    ('GenerateFeature',f2),
    ('preprocessor',preprocessor)
])

def transformed_data(X_train,X_test,Y_train):
    X_train_tranformed = full_pipe.fit_transform(X_train,Y_train)
    X_test_transformed = full_pipe.transform(X_test)
    return X_train_tranformed,X_test_transformed

