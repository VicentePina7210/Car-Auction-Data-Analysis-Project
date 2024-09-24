# Car-Auction-Data-Analysis-Project
A collaboration project that demonstrates proficiency with Python, NumPy, Pandas and general data science principles.

Initial Cleaning/Exploratory Analysis

For the cleaning of our project we found some problems within the dataset for example. The saledate was formatted in a way that showcased the date with the time and timezone. We figured that this was too much information, so instead we converted the format to just show the date itself. This data also had to be converted to the date time format instead of string for use in further analysis. There was also an issue where some rows were shifted one column over, resulting in errors, therefore these rows were dropped.

For the exploration component we mainly extracted important information such as average sale prices, the top selling vehicle brands, amount of cars with high mileage, percentage of cars in good condition, percentage of cars with automatic transmission, etc. Some interesting insights is that the lower milage cars and newer cars tended to sell for a higher price. However, there were some outliers where older regular cars like a Ford Escape was selling for 200,000. I believe this could be due to an error when listing the car for sale by adding and extra zero or something of that nature. 

Some of the key takeaways were as follows:

1. Selling price 
The average overall selling price was $13,690.40.
The top 10 most expensive cars include mostly high-end models like Ferrari 458 Italia ($183,000), Mercedes-Benz S-Class ($173,000), and multiple Rolls-Royce Ghost variants.
The distribution of selling prices was pretty varied because high-end models drove up the average.

2. Make and Model Analysis 
The most common car makes were Ford, Chevrolet, Nissan, Toyota, Dodge, Hyundai, BMW, Kia, and Chrysler.
High-end car brands like Ferrari, Rolls-Royce, Bentley, and Lambroghini had the highest average selling prices, averaging above   $100,000. Buick, Chevrolet, and Kia have lower averages, typically around $20,000. 

3. Transmission and Condition
96.54% of the cars had automatic transmission indicating most people prefer automatic.
64.62% of cars are sold in good condition correlating with a higher sale price. 

4. Sales Distribution
A substantial majority (435,547) were sold in 2015 compared with 2014 (36,789).
February and June are the months with the highest sales. 

5. Mileage and Age
There are 110,299 cars with above 100,000 miles.
The data has a wide variety of ages.

6. Geography
Tennessee, Colorado, and Nevada have the highest average sale prices while Alabama, North Carolina, and Virginia have lower averages. 
Tennessee, Colorado, and Illinois had the highest median price. 

Some interesting follow-up questions are:
1. Are there any brands or models that consistently outperform others in terms of holding their value over time?
2. What are the seasonal trends in car sales? Does the time of year impact the selling price or volume of sales?
3. Do certain models sell more frequently or for higher prices in certain states?
4. What factors cause cars to be sold more frequently in certain months?
5. Do certain cars sell better in certain months?

