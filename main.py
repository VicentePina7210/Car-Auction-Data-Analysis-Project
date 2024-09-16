import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

car_df = pd.read_csv("car_prices.csv", low_memory=False)
print(car_df.head(10))