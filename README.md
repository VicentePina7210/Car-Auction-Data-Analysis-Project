# Car-Auction-Data-Analysis-Project
A collaboration project that demonstrates proficiency with Python, NumPy, Pandas and general data science principles.

Initial Cleaning/Exploratory Analysis

For the cleaning of our project we found some problems within the dataset for example. The saledate was formatted in a way that showcased the date with the time and timezone. We figured that this was too much information, so instead we converted the format to just show the date itself. This data also had to be converted to the date time format instead of string for use in further analysis. There was also an issue where some rows were shifted one column over, resulting in errors, therefore these rows were dropped.

For the exploration component we mainly extracted important information such as average sale prices, the top selling vehicle brands, amount of cars with high mileage, percentage of cars in good condition, percentage of cars with automatic transmission, etc. Some interesting insights is that the lower milage cars and newer cars tended to sell for a higher price. However, there were some outliers where older regular cars like a Ford Escape was selling for 200,000. I believe this could be due to an error when listing the car for sale by adding and extra zero or something of that nature. 

Some interesting follow-up questions are:
1. Are there any brands or models that consistently outperform others in terms of holding their value over time?
2. What are the seasonal trends in car sales? Does the time of year impact the selling price or volume of sales?

