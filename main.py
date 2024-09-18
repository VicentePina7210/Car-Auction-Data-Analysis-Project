import pandas as pd
import numpy as np
import seaborn as sns
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
car_df['saledate'] = car_df['saledate'].dt.date
car_df['saledate'] = pd.to_datetime(car_df['saledate'], errors='coerce')


print(car_df.head(10))

print("\n\n\nInitial Exploratory Coding: \n")

# #######################First 10 Question###################################
# #What is the average selling price?
# price_avg = car_df['sellingprice'].mean()
# print(f"The average selling price is: {price_avg:.2f}")


# # Get the value counts of 'condition' and sort by ascending order
# condition_counts = car_df['condition'].value_counts().sort_index()

# # Plot the sorted data
# condition_counts.plot(kind='bar', title='Number of Cars by Condition')
# plt.xlabel('Condition')
# plt.ylabel('Count')
# plt.show()

# # What is the number of cars in each state?
# car_df['state'].value_counts().plot(kind='bar', title='Number of Cars by State')
# plt.xlabel('State')
# plt.ylabel('Count')
# plt.show()

# # What is the correlation between odometer and the selling price
# car_df_sampled = car_df[car_df['odometer'] < 500000].sample(5000, random_state=50)
# # Due to the dataset having about 500,000 rows the plot was too large, so a smaller sample size was used to increase readability
# car_df_sampled.plot.scatter(x='odometer', y = 'sellingprice', title = 'Odometer vs Selling Price')
# plt.xlabel('Odometer')
# plt.ylabel('Selling Price')
# plt.show()


# # What are the top 10 most expesnive selling cars?
# highest_sellers = car_df.nlargest(10, 'sellingprice')
# print(highest_sellers[['make','model', 'sellingprice']])
# # Most of these vehicle models make sense, however the ford escape being the highest price does not.


# # It can be useful to see the percentage of vehicles that have an automatic transmission
# transmission_percent = (car_df['transmission'] == 'automatic').mean()*100
# print(f"The percentage of automatic transmission cars is %{transmission_percent:.2f}")
# # This can be useful to determine if the non-automatic cars tend to be more or less expensive.

# # How many cars have been sold in each year?


# cars_per_year = car_df['saledate'].dt.year.value_counts()
# print(f"The amount of cars sold in each year is: \n{cars_per_year}")
# # Overall a lot more cars were sold in 2015 than 2014

# # What is the average price for each car make?
# price_avg_by_make = car_df.groupby('make')['sellingprice'].mean()
# print("The average selling price for each make is: \n")
# for make, price in price_avg_by_make.items():
#     print(f"{make}: ${price:,.2f}")
# # Rolls-Royce comes out as the most expensive selling make

# # What percent of cars were sold in good condition?
# good_condition_percent = (car_df['condition'] > 3.0).mean() * 100
# print(f"The percent of cars sold in a good condition (> 3.0) is: %{good_condition_percent:.2f}")

# # How many cars in the dataset have high mileage (>100,000)?
# high_mileage_cars = (car_df['odometer'] > 100000).sum()
# print(f"The amount of high mileage cars (miles > 100,000) in the set is: {high_mileage_cars}")

# #######################End of First 10 Questions###################################

# #######################Start of Middle 10 Questions##################################

# # What are the most common car makes in the dataset?
# make_counts = car_df['make'].value_counts().head(10) 
# print("The top 10 most common car makes are: ") 
# print(make_counts)

# # What is the distribution of car ages? 
# car_df['car_age'] = 2015 - car_df['year']
# car_df['car_age'].hist(bins = 20)
# plt.title('Distribution of Car Ages')
# plt.xlabel('Age (years)')
# plt.ylabel('Frequency')
# plt.show()

# # What is the average selling price per car model?
# price_avg_by_model = car_df.groupby('model')['sellingprice'].mean().sort_values(ascending=False)
# print("The average selling price per car model: ")
# print(price_avg_by_model.head(10))

# # What is the relationship between car age and selling price?
# car_df_sampled = car_df.sample(5000, random_state= 50)
# car_df_sampled.plot.scatter(x = 'car_age', y = 'sellingprice', title = 'Car Age vs. Selling Price')
# plt.xlabel('Car Age (years)')
# plt.ylabel('Selling Price')
# plt.show()

# # Which states have the highest selling price?
# avg_price_by_fuel = car_df.groupby('fuel')['sellingprice'].mean().sort_values(ascending = False)
# print("Average selling price by fuel type: ")
# print(avg_price_by_fuel)

# # How does the selling value vary by fuel type?
# avg_price_by_fuel = car_df.groupby('fuel')['sellingprice'].mean().sort_values(ascending = False)
# print("Average selling price by fuel type: ")
# print(avg_price_by_fuel)

# # What are the top 5 most common colors?
# color_counts = car_df['paintcolor'].value_counts().head(5)
# print("The top 5 most popular colors are:")
# print(color_counts)

# # How does the condition affect price?
# avg_price_by_condition = car_df.groupby('condition')['sellingprice'].mean()
# print('Average selling price by condition:')
# print(avg_price_by_condition)

# # How many cars are sold each month?
# car_df['sale_month'] = car_df['saledate'].dt.month
# cars_per_month = car_df['sale_month'].value_counts().sort_index()
# print("Number of cars sold each month:")
# print(cars_per_month)

# # What is the average mileage by make?
# car_df['sellingprice'].hist(bins = 50)
# plt.title('Distribution of Selling Prices')
# plt.xlabel('Selling Price')
# plt.ylabel('Frequency')
# plt.show()

# #######################End of Middle 10 Questions###################################

# #######################Start of Last 10 Questions###################################
# #What color car sells for the highest price? (Plotted) 
# #convert the column to a list
color_price = car_df.groupby('color')['sellingprice'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
color_price.plot(kind='bar', color='skyblue', width = 0.5)
plt.title('Average Selling Price by Car Color')
plt.xlabel('Car Color')
plt.ylabel('Average Selling Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()
#Does the condition of the car determine the selling price? 
condition_price = car_df.groupby('condition')['sellingprice'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,6))
condition_price.plot(kind = 'bar', color = 'skyblue', width = 0.5)
plt.xticks(rotation=45, ha='right')
plt.show()

#Do states prefer a certain color? 
state_color_counts = car_df.groupby(['state', 'color']).size().unstack(fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(state_color_counts, cmap='Blues', annot=False, linewidths=0.5)

plt.title('Car Color Counts by State')
plt.xlabel('Car Color')
plt.xticks(rotation=45, ha='right')
plt.ylabel('State')

# Show the heatmap
plt.show()
#What month sells the most cars? 
car_df['month'] = pd.to_datetime(car_df['saledate']).dt.month

most_sales_month = car_df['month'].value_counts().idxmax()
print("the state with the most sales is: ")
print(most_sales_month)
#this needs to be converted to string

#What state has the highest mileage cars sold on average?
highest_mileage_state = car_df.groupby('state')['odometer'].mean().idxmax()
print("the state with the highest mileage cars sold is: ")
print(highest_mileage_state)
#What year car is most common?
most_common_year = car_df['year'].value_counts().idxmax()
print("the most common year is: ")
print(most_common_year)
#What make sells best overall?
best_selling_make = car_df['make'].value_counts().idxmax()
print("the best selling make is: ")
print(best_selling_make)

#What seller makes up the most sales?
top_seller = car_df['seller'].value_counts().idxmax()
print("the top seller is: ")
print(top_seller)

#What is the average condition of the cars? 
average_condition = car_df['condition'].mean()
print("the average condition is: ")
print(average_condition)

#######################End of Last 10 Questions###################################