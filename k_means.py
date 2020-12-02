import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import DistanceMetric
import random
import matplotlib.pyplot as plt

# Simulate dataframe
x1 = np.random.uniform(low=0, high=2500, size=(200))
x2 = np.random.uniform(low=5, high=3000, size=(200))

df_pd = pd.DataFrame({'x1': x1, 'x2': x2}, columns=['x1', 'x2'])

# Compute distances
dist = DistanceMetric.get_metric('euclidean')  # euclidean distance
df = df_pd[['x1', 'x2']].to_numpy()

distance = dist.pairwise(df)  # matrix with distances

# Get the edges in ascending order
sorted_edges = np.transpose(np.unravel_index(np.argsort(distance, axis=None), distance.shape)).tolist()
sorted_edges= sorted_edges[200::2] # Remove the first< 200 as they are 0 and as they are duplicated, choose one for each 2

# Sorted distances
sorted_dist = np.sort(distance, axis=None).tolist()
sorted_dist = sorted_dist[200::2]

df_pd['edge'] = df_pd[['x1', 'x2']].apply(tuple, axis=1)
df_pd['cluster'] = ['Non visited' for i in range(0,200)]


def centeroid(array):
    length = len(array)
    sum_x = sum(i for i, j in array)
    sum_y = sum(j for i, j in array)
    return sum_x/length, sum_y/length


# Algorithm

def k_means(num, df_pd):
    final_clusters = df_pd.iloc[random.choices(df_pd.index.values, k=num), 2].values.tolist()
    clus_original = df_pd['cluster'].tolist()
    clus_final = [0] * len(clus_original)

    while clus_original != clus_final:

        initial_clusters = final_clusters
        clus_original = df_pd['cluster'].tolist()

        for edge in range(0, len(df_pd)):
            distances_to_cluster = {'cluster': initial_clusters, 'distance': []}

            for i in initial_clusters:
                dist = np.linalg.norm(np.array(df_pd.iloc[edge, 2]) - np.array(i))
                distances_to_cluster['distance'] += [dist]

            dist_edge = pd.DataFrame(distances_to_cluster)
            new_clus = (dist_edge.loc[(dist_edge['distance'] == dist_edge['distance'].min()), 'cluster']).values

            df_pd.iloc[edge, 3] = new_clus

        final_clusters = []
        for i in df_pd.cluster.unique():
            final_clusters += [centeroid(df_pd[df_pd['cluster'] == i]['cluster'].tolist())]

        clus_final = df_pd['cluster'].tolist()

    return df_pd

cl = k_means(3, df_pd)

sns.lmplot( x="x1", y="x2", data=cl, fit_reg=False, hue='cluster', legend = True)
plt.show()