# Prediction Model

#### Problem

Company wants to automate the loan eligibility process based on customer detail provided while filling online application form.
It is a classification problem where we have to predict whether a loan would be approved or not.

#### Data

The data corresponds to a set of financial transactions associated with individuals. The data has been standardized, de-trended, and anonymized.

| Variables         | Description                                    |
| ----------------- | ---------------------------------------------- |
| Loan_ID           | Unique Loan ID                                 |
| Gender            | Male/ Female                                   |
| Married           | Applicant married (Y/N)                        |
| Dependents        | Number of dependents                           |
| Education         | Applicant Education (Graduate/ Under Graduate) |
| Self_Employed     | Self employed (Y/N)                            |
| ApplicantIncome   | Applicant income                               |
| CoapplicantIncome | Coapplicant income                             |
| LoanAmount        | Loan amount in thousands                       |
| Loan_Amount_Term  | Term of loan in months                         |
| Credit_History    | credit history meets guidelines                |
| Property_Area     | Urban/ Semi Urban/ Rural                       |
| Loan_Status       | Loan approved (Y/N)                            |

Source: Kaggle

## Developement mode

Go to the project directory and install dependencies

```python
pip install -r requirements.txt
```

Create a pickle file

```python
python prediction_model/train_pipeline.py
```

Creating a source distribution and wheel

```python
python setup.py sdist bdist_wheel
```

## Running Tests

To run tests, run the following command

```bash
  pytest -v
```

This will look for `test_*.py` or `*_test.py` files into directories and sub-directories

```python
============================= test session starts ==============================
platform linux -- Python 3.6.9, pytest-4.6.11, py-1.10.0, pluggy-0.13.1 -- /home/suhas/code/venv_package/bin/python
cachedir: .pytest_cache
rootdir: /home/suhas/code/packages
collected 3 items

tests/test_predict.py::test_single_prediction_not_none PASSED            [ 33%]
tests/test_predict.py::test_single_prediction_dtype PASSED               [ 66%]
tests/test_predict.py::test_single_prediction_output PASSED              [100%]

=========================== 3 passed in 1.27 seconds ===========================
```
