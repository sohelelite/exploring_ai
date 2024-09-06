import pathlib
import os
import prediction_model
import sys

print("Module file:", prediction_model.__file__)
print("Module name:", prediction_model.__name__)
print("Sys path:", sys.path)

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent


DATAPATH = os.path.join(PACKAGE_ROOT, 'datasets')
SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

TARGET = 'Loan_Status'

#Features to keep
FEATURES = [
                'Gender',
                'Married',
                'Dependents',
                'Education',
                'Self_Employed',
                'ApplicantIncome',
                'CoapplicantIncome',
                'LoanAmount',
                'Loan_Amount_Term',
                'Credit_History',
                'Property_Area'
            ]

NUMERICAL_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CATEGORICAL_FEATURES = [
                'Gender',
                'Married',
                'Dependents',
                'Education',
                'Self_Employed',
                'Credit_History',
                'Property_Area'
                ]

FEATURES_TO_ENCODE = [
                'Gender',
                'Married',
                'Dependents',
                'Education',
                'Self_Employed',
                'Credit_History',
                'Property_Area'
                ]

TEMPORAL_FEATURES = ['ApplicantIncome']

TEMPORAL_ADDITION = ['CoapplicantIncome']

LOG_FEATURES = ['ApplicantIncome', 'LoanAmount']

DROP_FEATURES = ['CoapplicantIncome']