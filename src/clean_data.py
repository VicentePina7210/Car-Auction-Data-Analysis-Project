import pandas as pd
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
import os


raw_data_path = os.path.join('data', 'raw', 'car_prices.csv')
cleaned_df = pd.read_csv(raw_data_path, low_memory=False)
# Dropping the last empty column
non_null = cleaned_df.iloc[:,-1].dropna()

#Because the data set is very large we can afford to drop the values that are null
#first we drop the last column because it only contains 25 non nulls
cleaned_df = cleaned_df.drop(cleaned_df.columns[-1], axis = 1)
cleaned_df['condition'] = pd.to_numeric(cleaned_df['condition'], errors='coerce')
cleaned_df = cleaned_df.dropna()

# The saledate column has a format which contains too much information, it is better to convert this to a more standard format.
# An error was encountered due to some dates having years out of normal range
cleaned_df['saledate'] = pd.to_datetime(cleaned_df['saledate'], errors='coerce', utc=True)
print(cleaned_df['saledate'].isnull().sum())
# Now convert to standard format
cleaned_df['saledate'] = cleaned_df['saledate'].dt.date
cleaned_df['saledate'] = pd.to_datetime(cleaned_df['saledate'], errors='coerce')

cleaned_data_path = os.path.join('data', 'cleaned', 'car_prices_cleaned.csv')
cleaned_df.to_csv(cleaned_data_path, index=False)

