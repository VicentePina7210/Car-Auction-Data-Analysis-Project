import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

car_df = pd.read_csv("car_prices.csv", low_memory=False)
# First it is important to get an idea of how many null values there are within the dataset
print(car_df.info())

print(car_df.iloc[:, -1].head(5))

non_null = car_df.iloc[:,-1].dropna()

print(non_null)

#Because the data set is very large we can afford to drop the values that are null
#first we drop the last column because it only contains 25 non nulls
car_df = car_df.drop(car_df.columns[-1], axis = 1)
car_df['condition'] = pd.to_numeric(car_df['condition'], errors='coerce')
car_df = car_df.dropna()

print(car_df.info())

#It seems that the last column might be unnecessary as it is unnamed and mostly filled with null values

pd.set_option('display.max_columns', None)

# The saledate column has a format which contains too much information, it is better to convert this to a more standard format.
# An error was encountered due to some dates having years out of normal range
car_df['saledate'] = pd.to_datetime(car_df['saledate'], errors='coerce', utc=True)
print(car_df['saledate'].isnull().sum())
# Now convert to standard format
car_df['saledate'] = pd.to_datetime(car_df['saledate'])
car_df['saledate'] = car_df['saledate'].dt.date


print(car_df.head(10))
