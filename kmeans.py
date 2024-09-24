#from main import car_df
import pandas as pd
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def prep_data():
    car_df = pd.read_csv("car_prices.csv", low_memory=False)
    car_df = car_df.drop(car_df.columns[-1], axis = 1)
    car_df['condition'] = pd.to_numeric(car_df['condition'], errors='coerce')
    car_df = car_df.dropna()
    car_df['saledate'] = pd.to_datetime(car_df['saledate'], errors='coerce', utc=True)
    print(car_df['saledate'].isnull().sum())
    # Now convert to standard format
    car_df['saledate'] = car_df['saledate'].dt.date
    car_df['saledate'] = pd.to_datetime(car_df['saledate'], errors='coerce')
    return car_df

df = prep_data()

class CustomKMeans:
    def __init__(self, k, max_iters = 500, tol = 1e-4, random_state = None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.inertia_ = None
    
    def fit(self):
        '''
        km = KMeans(n_clusters = self.k)
        km = km.fit(self.df)
        return km
        '''
        data = df.values
        n_samples, n_features = data.shape

        self.centroids = self.initialize_centroids(data)

        for i in range(self.max_iters):
            # assign clusters
            labels = self.compute_distance(data, self.centroids)
            # update centroids
            new_centroids = self.update_centroids(data,labels)
            
            # check for convergence
            centroid_shifts = np.linalg.norm(self.centroids - new_centroids, axis = 1)
            if np.all(centroid_shifts <= self.tol):
                print(f"converged at iteration {i + 1}")
                break

            self.centroids = new_centroids
            
    # randomly select points for each centroid
    def initialize_centroids(self, data):
        #This is meant retreive the row numbers of the datafram (data.index)
        #The code np.random.choice is what allows for a random element to be selected from the afformentioned index
        #The parameter replace=False allows for selection of unique rows
        random_centroids = np.random.choice(data.index, self.k, replace=False)
        return df.loc[random_centroids].to_numpy()
        # This accesses the rows that were generated previously using the random centroids
        #The function .to_numpy() returns this as numpy array for further use
    
    
    # assign each point by checking the nearest centroid
    def compute_distance(self, data, centroids):
        return None
    
    # update centroids based on the average of all data points assigned to the cluster
    def update_centroids(self, data, labels):
        return None
    
    def predict(self):
        '''
        cluster_ids = self.fit().predict(self.df)
        return cluster_ids
        '''
        return None
        

    def sse(self):
        '''
        km = self.fit()
        inertia = km.inertia_
        return inertia
        '''
        return None



