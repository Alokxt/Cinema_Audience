from src.data.load import load_data
import pandas as pd 
import numpy as np  

booknow_booking,booknow_theaters,cinepos_booking,cinepos_theater,booknow_visits,relation_id,date_info = load_data()
print(booknow_booking.info())

#>No missing value in Booknow_booking

print(cinepos_booking.isnull().sum())

#->No missing value in cinepos_booking

print(booknow_theaters.isnull().sum())

#->Theater id are missing,we cannot impute this with anything , so we drop this

def get_processed_data():
    booknow_booking['show_date'] = pd.to_datetime(booknow_booking['show_datetime'].apply(lambda x:x[:10]))
    cinepos_booking['show_date'] = pd.to_datetime(cinepos_booking['show_datetime'].apply(lambda x:x[:10]))        
    booknow_visits['show_date'] = pd.to_datetime(booknow_visits['show_date'])

    booknow_visits['week_day'] = booknow_visits['show_date'].apply(lambda x:x.dayofweek)
    booknow_visits['month'] = booknow_visits['show_date'].apply(lambda x:x.month)
    booknow_visits['year'] = booknow_visits['show_date'].apply(lambda x:x.year)
    booknow_visits['day'] = booknow_visits['show_date'].apply(lambda x:x.dayofyear)

    booknow_booking2 = booknow_booking.groupby(['book_theater_id','show_date'],as_index=False).agg({'tickets_booked':'sum'})
    cinepos_booking2 = cinepos_booking.groupby(['cine_theater_id','show_date'],as_index=False).agg({'tickets_sold':'sum'})

    booknow_booking2['week_day'] = booknow_booking2['show_date'].apply(lambda x:x.dayofweek)
    booknow_booking2['month'] = booknow_booking2['show_date'].apply(lambda x:x.month)
    cinepos_booking2['week_day'] = cinepos_booking2['show_date'].apply(lambda x:x.dayofweek)
    cinepos_booking2['month'] = cinepos_booking2['show_date'].apply(lambda x:x.month)

    return booknow_booking2,booknow_theaters,cinepos_booking2,cinepos_theater,booknow_visits,relation_id,date_info



#->longitude and latitude of cinepos are imputed with mean

