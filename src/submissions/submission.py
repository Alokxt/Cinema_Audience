from src.data.load import get_test_data
from src.models.experiments import get_final_model
from src.models.experiments import predict_test_data

model = get_final_model()


def submit_predictions():
    X_test = get_test_data()
    preds = predict_test_data(X_test,model,0,0,0)

    X_test['audience_count'] = preds 
    X_test.to_csv('submission.csv',index=False)
    return 

