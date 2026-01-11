from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder,LabelEncoder 
import pandas as pd 
import numpy as np 
from src.preprocessing.data_preprocessing import get_processed_data



booknow_booking,booknow_theaters,cinepos_booking,cinepos_theater,booknow_visits,relation_id,date_info  = get_processed_data()


class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, booknow_theater, 
                 weekday_tcktsbook, weekday_tcktsold,
                 month_tcktsold, month_tcktsbooked,mean_audiences):
        
        self.booknow_theater = booknow_theater
        
        self.mean_audiences = mean_audiences
        
        
        self.weekday_tcktsbook = weekday_tcktsbook
        self.weekday_tcktsold = weekday_tcktsold
        self.month_tcktsold = month_tcktsold
        self.month_tcktsbooked = month_tcktsbooked
        self.fitted = False

    def fit(self, X, y):
        self.fitting = True 
        self.fitted = True 
        self.training_data = X.copy()
        self.training_data['audience_count'] = y
       
        
         
        
        return self
    def find_mean_aud(self,x):
        d = self.mean_audiences[self.mean_audiences['book_theater_id'] == x]
        if d.shape[0]>0:
            return d['audience_count'].iloc[0]
        return 30
        
    def tckts_book_week(self,x):
        return self.weekday_tcktsbook.iloc[x]['tickets_booked']
    
    def tckts_sold_week(self,x):
        return self.weekday_tcktsold.iloc[x]['tickets_sold']
        
    def tckts_sold_month(self,x):
        return self.month_tcktsold.iloc[x]['tickets_sold']
        
    def tckts_book_month(self,x):
        return self.month_tcktsbooked.iloc[x]['tickets_booked'] 
            
        

    def transform(self, data):
        
        if self.fitted == False:
            return "this did not fitted"

        X= pd.merge(data,self.booknow_theater,on='book_theater_id',how='left')
        date = pd.to_datetime(X["date"])
        X["year"] = date.dt.year
        X["month"] = date.dt.month
        X["week_day"] = date.dt.dayofweek
        X["day"] = date.dt.dayofyear

        X["is_weekend"] = X["week_day"].isin([5, 6, 0]).astype(int)
        X["is_dec"] = (X["month"] == 12).astype(int)
        X["is_summer"] = X["month"].isin([3, 4, 5, 6]).astype(int)

        X["mean_audience"] = X["book_theater_id"].apply(lambda x:self.find_mean_aud(x))
        X["tkt_sold_week"] = X["week_day"].apply(self.tckts_sold_week)
        X["tkt_sold_month"] = (X["month"] - 1).apply(self.tckts_sold_month)
        X["tkt_book_week"] = X["week_day"].apply(self.tckts_book_week)
        X["tkt_book_month"] = (X["month"] - 1).apply(self.tckts_book_month)
        X["prior_info"] = 0.0
        X["lag1"] = 0.0
        X["lag7"] = 0.0
        if self.fitting == True:
            X["prior_info"] = self.training_data.groupby('book_theater_id')['audience_count'].expanding().mean().shift(1).reset_index(level=0, drop=True)
            X["lag1"] = self.training_data['audience_count'].shift(1)
            X["lag7"] = self.training_data['audience_count'].shift(7)
            
            self.fitting = False
       
        X = X.drop(columns=['book_theater_id','date','theater_area','month'])
        
        return X


def get_feature_data():
    mean_audiences = booknow_visits.groupby('book_theater_id',as_index=False).agg({'audience_count':'mean'})
    weekday_tcktsold = cinepos_booking.groupby('week_day').agg({'tickets_sold':'mean'}).astype(int).reset_index()
    weekday_tcktsbook = booknow_booking.groupby('week_day').agg({'tickets_booked':'mean'}).astype(int).reset_index()
    month_tcktsold = cinepos_booking.groupby('month').agg({'tickets_sold':'mean'}).astype(int).reset_index()
    month_tcktsbooked = booknow_booking.groupby('month').agg({'tickets_booked':'mean'}).astype(int).reset_index()
    return booknow_theaters,weekday_tcktsbook, weekday_tcktsold,month_tcktsold, month_tcktsbooked,mean_audiences



