# Credit Card Transaction Fraud Detection

## Project Overview

The objective of this project is to build Machine Learning models that can accurately detect fraudulent transactions within a highly imbalanced dataset. The dataset consists of transaction records labelled as fraud (`1`) or non-fraud (`0`).

## Dataset Variables

- `Unnamed: 0`: Index column (to be removed)
- `trans_date_trans_time`: Transaction DateTime
- `cc_num`: Credit Card Number of Customer
- `merchant`: Merchant Name
- `category`: Category of Merchant
- `amt`: Amount of Transaction
- `first`: First Name of Credit Card Holder
- `last`: Last Name of Credit Card Holder
- `gender`: Gender of Credit Card Holder
- `street`: Street Address of Credit Card Holder
- `city`: City of Credit Card Holder
- `state`: State of Credit Card Holder
- `zip`: Zip Code of Credit Card Holder
- `lat`: Latitude Location of Credit Card Holder
- `long`: Longitude Location of Credit Card Holder
- `city_pop`: Credit Card Holder's City Population
- `job`: Job of Credit Card Holder
- `dob`: Date of Birth of Credit Card Holder
- `trans_num`: Transaction Number
- `unix_time`: UNIX Time of Transaction
- `merch_lat`: Latitude Location of Merchant
- `merch_long`: Longitude Location of Merchant
- `is_fraud`: Fraud Flag (Target Class)

## Navigating the repository

- `notebooks/00_eda.ipynb`: Initial Exploratory Data Analysis of dataset
- `notebooks/01_sampling.ipynb`: Resampling & Dataset Splitting
- `notebooks/02_preprocessing_fe.ipynb`: Data Preprocessing & Feature Engineering
- `notebooks/03_modelling.ipynb`: Model Building, Tuning, and Evaluation
- `notebooks/model_evaluator.py`: Custom evaluator module for model evaluation
- `data/raw`: To contain the original datasets from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- `data/preliminary`: To contain the datasets produced after Resampling & Dataset Splitting (`notebooks/01_sampling.ipynb`)
- `data/processed`: To contain the datasets produced after Data Preprocessing & Feature Engineering (`notebooks/02_preprocessing_fe.ipynb`)

## Installing Dependencies

1. Clone this repository and navigate to the root directory
2. Create a Python virtual environment and activate it
3. Install the required Python packages using the following command:
   ```
   pip install -r requirements.txt
   ```
4. Add the virtual environment to Jupyter using the following command:
   ```
   python -m ipykernel install --name=myenv
   ```
