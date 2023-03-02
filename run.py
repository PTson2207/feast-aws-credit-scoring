import boto3
import pandas as pd

from credit_model import CreditScoringModel

#load history data
loans = pd.read_parquet("data/loan_table.parquet")

model = CreditScoringModel()

#train model using Redshift for zopcode and credit hisroty feature
if not model.is_model_trained():
    model.train(loans)

#create data sample to prediction online(using DynamoDB for retriveing online feature)
loan_request = {
    "zipcode": [76104],
    "dob_ssn": ["19630621_4278"],
    "person_age": [133],
    "person_income": [59000],
    "person_home_ownership": ["RENT"],
    "person_emp_length": [123.0],
    "loan_intent": ["PERSONAL"],
    "loan_amnt": [35000],
    "loan_int_rate": [16.02],
}

result = model.predict(loan_request)

if result == 0:
    print("Loan approved!")
elif result == 1:
    print("Loan rejected!")