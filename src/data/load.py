import pandas as pd 
import numpy as np 

booknow_theaters = pd.read_csv('src\\data\\booknow_theaters.csv')
booknow_booking = pd.read_csv('src\\data\\booknow_booking.csv')
cinepos_theater = pd.read_csv('src\\data\\cinepos_theaters.csv')
cinepos_booking = pd.read_csv('src\\data\\cinepos_booking.csv')
booknow_visits = pd.read_csv('src\\data\\booknow_visits.csv')
date_info = pd.read_csv('src\\data\\date_info.csv')
relation_id = pd.read_csv('src\\data\\relation_id.csv')
sample_sub = pd.read_csv('src\\data\\sample_sub.csv')

# convert dates from object type to datetime

#print(booknow_booking.head())

def load_data():
    booknow_booking['show_date'] = pd.to_datetime(booknow_booking['show_datetime'].apply(lambda x:x[:10]))
    cinepos_booking['show_date'] = pd.to_datetime(cinepos_booking['show_datetime'].apply(lambda x:x[:10]))        
    booknow_visits['show_date'] = pd.to_datetime(booknow_visits['show_date'])

    return booknow_booking,booknow_theaters,cinepos_booking,cinepos_theater,booknow_visits,relation_id,date_info

def get_test_data():
    sample_sub["book_theater_id"] = sample_sub["ID"].apply(lambda x:x[:10])
    sample_sub["date"] = sample_sub["ID"].apply(lambda x:x[11:22])
    sample_sub = sample_sub.drop(columns=['ID','audience_count'])

    return sample_sub

