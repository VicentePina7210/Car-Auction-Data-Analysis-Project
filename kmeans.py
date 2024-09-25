#from main import car_df
import pandas as pd
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
'''
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
df = df.select_dtypes(include=[np.number])
'''
# TEST 
# this line only includes numeric columns since Kmeans won't work with non numeric stuff

class CustomKMeans:
    def __init__(self, k, max_iters = 500, tol = 1e-4, random_state = None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.inertia_ = None
    
    def fit(self, df):
        '''
        km = KMeans(n_clusters = self.k)
        km = km.fit(self.df)
        return km
        '''
        data = df.values  # this extracts the underlying NumPy array from the dataframe
        # data.shape returns a tuple for the dimensions of the array
        n_samples, n_features = data.shape  #n_samples represents the number of rows and n_features the number of columns 

        self.centroids = self.initialize_centroids(data)  # randomly selects k data points to serve as initial centroids

        for i in range(self.max_iters):
            # assign clusters
            labels = self.compute_distance(data, self.centroids) # finds the nearest centroid for each data point
            # update centroids
            new_centroids = self.update_centroids(data,labels) # stores new centroid spots after recalculating
            
            # check for convergence
            # this linalg.norm calculates this thing called euclidean distance for each shift
            centroid_shifts = np.linalg.norm(self.centroids - new_centroids, axis = 1) # centroid shifts shows how much each element has moved in the iteration
            if np.all(centroid_shifts <= self.tol): # this checks if all centroid shifts are less than or equal to the tolerance of self.tol
                # in other words, this is checking if the algorithm has converged by ensuring no centroids are moving too much
                print(f"converged at iteration {i + 1}") # prints out when there is a convergence
                break

            self.centroids = new_centroids # updates the instance variable centroids with the newly calculated ones
        self.inertia_ = self.sse(data, self.centroids, labels) # this assigns the final inertia value 
        # this calculated sthe sum of squared errors to measure the compactness of clusters
            
    # randomly select points for each centroid
    def initialize_centroids(self, data):
        #This is meant retreive the row numbers of the datafram (data.index)
        #The code np.random.choice is what allows for a random element to be selected from the afformentioned index
        #The parameter replace=False allows for selection of unique rows
        '''
        random_centroids = np.random.choice(data.index, self.k, replace=False)
        return df.loc[random_centroids].to_numpy()
        '''
        if self.random_state: # checks if a random_state (like a seed value) has been provided 
            np.random.seed(self.random_state) # this sets the seed for numPy's random number generator 
        
        # data.shape returns the shape of the array as a tuple
        # self.k grabs the number of clusters to decide how many centroid values to calculate
        # replace = false makes sure that each selected index is unique 
        random_indices = np.random.choice(data.shape[0], self.k, replace = False)

        # this selects the data points from the NumPy array that correspond to the randomly chosen indices
        centroids = data[random_indices]

        return centroids
    
    # assign each point by checking the nearest centroid
    def compute_distance(self, data, centroids):

        # data[:, np.nexaxis] is to add a new index to the data array to the data array, this makes it (n_samples, 1, n_features)
        # instead of being (n_samples, n_features) then
        # data[:, np.newaxis] - centroids subtracts each centroid from each data point
        # NumPy broadcasts centroids across the new axis where the subtraction yields an array like (n_samples, n_clusters, n_features)
        # each element there [i, j, :] represents the vector difference between i and j 
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis = 2)

        # this computes the Euclidean norm along the specified axis
        labels = np.argmin(distances, axis = 1)
        return labels
    
    # update centroids based on the average of all data points assigned to the cluster
    def update_centroids(self, data, labels):
        # create a new numpy array to hold the updated centroids
        # for k in range(self.k) loops over each cluster index 
        # data[labels == k] extracts all data points assigned to each cluster
        # np.any(labels == k) determines if there are any data points assigned to the cluster k
        # data[labels == k].mean(axis = 0) calculates the mean of selected points on each feature and calculates coordinates for cluster
        # else self.centroids[k] if no data points are assigned to cluster k, retains existing centroid spot
        new_centroids = np.array([data[labels == k].mean(axis = 0) if np.any(labels == k) else self.centroids[k]
                                  for k in range(self.k)
                                  ])
        return new_centroids
    
    def predict(self, df):
        '''
        cluster_ids = self.fit().predict(self.df)
        return cluster_ids
        '''
        data = df.values  # converts the panda df into a numpy array
        labels = self.compute_distance(data, self.centroids)  # determines the nearest centroid for each data point
        return labels
        

    def sse(self, data, centroids, labels):
        '''
        km = self.fit()
        inertia = km.inertia_
        return inertia
        '''
        inertia = 0  # sets the inertia to 0 initially
        for k in range(self.k): 
            cluster_data = data[labels == k] # extracts all data points assigned to cluster with value k
            if cluster_data.size == 0:  # if the cluster has no data points it skips
                continue
            # this line gets the euclidean distance of data points in cluster k to centroid k 
            distances = np.linalg.norm(cluster_data - centroids[k], axis = 1) # subtracts the centroid coordinates of cluster k from each data point
            inertia += np.sum(distances ** 2) # squares each elemnent and then sums the distances to add to inertia
        return inertia 


'''
kmeans_custom = CustomKMeans(k = 3, max_iters = 100, tol = 1e-4, random_state= 42)
kmeans_custom.fit(df)

labels = kmeans_custom.predict(df)
'''


def plot_clusters(df, labels, centroids, x_col, y_col):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.5)
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    
    plt.title('KMeans Clustering')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.grid(True)
    plt.show()

'''

plot_clusters(df, labels, kmeans_custom.centroids, 'odometer', 'sellingprice')


kmeans_sklearn = KMeans(n_clusters = 3, max_iter= 100, tol= 1e-4, random_state=42)
kmeans_sklearn.fit(df)
labels_sklearn = kmeans_sklearn.predict(df)

print(f"Scikit-learn kmeans inertia: {kmeans_sklearn.inertia_}")
print(f"custom kmeans inertia: {kmeans_custom.inertia_}")
'''


'''
Basically our charts are coming out really janky because this only works on numeric values and the only straight numeric
columns are year, condition, odometer, and sellingprice. those columns don't seem to like to cluster well. 
'''