import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial import distance
from sklearn.neighbors import DistanceMetric
import random
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_blobs


# Function to pick the closest cluster c to the point x
def closest(x, clusters):
    c0 = []
    d0 = np.Inf

    for c in clusters:
        d = np.linalg.norm(c - x)  # funtction to compute the distance between the cluster c and the point x
        if d < d0:
            c0 = c
            d0 = d
    return c0


# Function to see two points are approximately the same (if its distance is below 0.000000001)
def equal_clusters(c1, c2):
    for c in range(len(c1)):
        if np.linalg.norm(c1[c] - c2[c]) > 0.000000001:
            return False
    return True


# Algorithm
def k_means(path, k):

    df_aux = pd.read_csv(path)
    df_pd = pd.DataFrame() # create an empty pandas data frame

    # Add to the df a column with the point coordinates
    df_pd['point'] = df_aux.apply(np.array, axis=1)

    # Set the initial clusters as random points of the df
    clusters = df_pd.sample(k)['point'].to_list()

    # Create a column cluster with the coordinated of the cluster with minimum distance to each point
    df_pd['cluster'] = df_pd.apply(lambda x: closest(x.point, clusters), axis=1)

    # Set the final_cluster as the assigned clusters to all the points and the initial as the points (they're different)
    initial_clustering = np.array(df_pd['point'])
    final_clustering = np.array(df_pd['cluster'])

    # Keep iterating until the final clusters are equal to initial clusters
    while not equal_clusters(final_clustering, initial_clustering):

        # Set the initial clusters to be the final clusters of the last iteration
        initial_clustering = final_clustering

        # Calculate the centroids of the cluster as the mean of the points' coordinated belonging to that same cluster
        df_pd['tuple'] = df_pd['cluster'].map(tuple)

        def f(x):
            return x.mean()

        clusters = [np.round(v, 3) for v in df_pd.groupby('tuple')['point'].apply(lambda x: x.mean()).values]

        # Assign points the closest centroids to the points in the cluster column
        df_pd['cluster'] = df_pd.apply(lambda x: closest(x.point, clusters), axis=1)

        # Finally assing final clusters as the new clusters
        final_clustering = np.array(df_pd['cluster'])

    return df_pd


# Function to compute the DB index
def db_k_means(cl_k):
    db = pd.DataFrame(cl_k['point'].to_list(), columns=[i for i in range(0, len(cl_k['point'].to_list()[0]))])
    db['cluster'] = cl_k['cluster']

    db_in = sklearn.metrics.davies_bouldin_score(db.drop(['cluster'], axis=1), [str(i) for i in db['cluster']])

    return db_in


# Function to compute the Dunn index
def dunn_index(df):
    df['tuple'] = df['cluster'].map(tuple)  # tuple of the coordinate
    clusters = df.groupby(df["tuple"])

    big_delta = []
    small_delta = []

    for i in clusters.groups.keys():
        for k in df["point"][clusters.groups[i]]:
            for m in df["point"][clusters.groups[i]]:
                big_delta.append(np.linalg.norm(np.array(k)-np.array(m)))
        s1 = list(df["point"][clusters.groups[i]])
        for j in clusters.groups.keys():
            s2 = list(df["point"][clusters.groups[j]])
            small_delta_prov = distance.cdist(s1,s2).min(axis=1)
            small_delta.append(min(small_delta_prov))
        small_delta = list(filter(lambda x: x!=0,small_delta))

    return min(small_delta)/max(big_delta)





#if __name__ == '__main__':
   # k_num = 15
    # seed = 8
    #    class_data = make_blobs(n_samples=[100, 200, 150], n_features=2, random_state=seed)
    #    df_aux = pd.DataFrame()
    #    df_aux = pd.DataFrame(class_data[0])

   # df_aux = pd.read_csv('/Users/ividalquadras/Desktop/synthetic_clean.csv')

  #  df_pd = pd.DataFrame()
  #  df_pd['point'] = df_aux.apply(np.array, axis=1)

   # cl = k_means(df_pd, k_num)

   # pl = pd.DataFrame({'x1': [v[0] for v in cl['point']], 'x2': [v[1] for v in cl['point']],
     #                  'cluster': [str(i) for i in cl['tuple']]})
   # sns.lmplot(x="x1", y="x2", data=pl, fit_reg=False, hue='cluster', legend=False)
   # plt.show()
