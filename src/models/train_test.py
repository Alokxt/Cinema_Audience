from sklearn.model_selection import train_test_split
from src.data.load import load_data

from src.preprocessing.final_pipeline import transformed_data
def get_train_test_data():
    _,_,_,_,booknow_visits,_,_ = load_data()
    booknow_visits = booknow_visits.rename(columns={'show_date':'date'})
    X = booknow_visits.drop(columns='audience_count')
    y = booknow_visits['audience_count']

    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

    X_train_transformed,X_test_transformed = transformed_data(X_train,X_test,Y_train)

    return X_train_transformed,Y_train,X_test_transformed,Y_test




