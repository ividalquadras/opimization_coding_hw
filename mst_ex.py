import numpy as np
from numpy import random
import pandas as pd
from sklearn.neighbors import DistanceMetric
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_blobs


class_data = pd.read_csv('/Users/ividalquadras/Desktop/synthetic_clean.csv')

df_pd = pd.DataFrame(class_data)
df_pd.columns = ['x1', 'x2']

def sort(df_pd):
    l = len(df_pd)
    df_pd['cluster'] = ['Non visited' for i in range(0,l)]

    # Compute distances
    dist = DistanceMetric.get_metric('euclidean')
    df = df_pd[['x1', 'x2']].to_numpy()

    distance = dist.pairwise(df)  # matrix with distances


    # Get the edges in ascending order
    sorted_edges = np.transpose(np.unravel_index(np.argsort(distance, axis=None), distance.shape)).tolist()
    sorted_edges = sorted_edges[l::2] # Remove the first 0 and as they are duplicated, choose one for each 2

    # Sorted distances
    sorted_dist = np.sort(distance, axis=None).tolist()
    sorted_dist = sorted_dist[l::2]

    return (sorted_edges, sorted_dist, df_pd)

# Define the algorithm

def get_mst(k, N, sorted_edges, sorted_dist):

    sorted_edges = sort(df_pd)[0]


    cluster_dic = {-1: []}
    counter = 0

    for edge in sorted_edges:

        parent_0 = df_pd.iloc[edge[0], 2]
        parent_1 = df_pd.iloc[edge[1], 2]

        if (parent_0 == parent_1) and (parent_0 != 'Non visited'):
            pass

        elif (parent_0 == 'Non visited') and (parent_1 == 'Non visited'):
            counter += 1
            max_cluster = max(cluster_dic)
            cluster_dic[max_cluster + 1] = [edge[0]]
            cluster_dic[max_cluster + 1] += [edge[1]]

            df_pd.iloc[edge[0], 2] = max_cluster + 1
            df_pd.iloc[edge[1], 2] = max_cluster + 1


        elif (parent_0 == 'Non visited') or (parent_1 == 'Non visited'):
            counter += 1
            if parent_0 == 'Non visited':
                cluster_dic[parent_1] += [edge[0]]
                df_pd.iloc[edge[0], 2] = parent_1
            else:
                cluster_dic[parent_0] += [edge[1]]
                df_pd.iloc[edge[1], 2] = parent_0

        else:
            counter += 1
            cluster_dic[parent_0] += cluster_dic[parent_1]
            del cluster_dic[parent_1]

            df_pd.iloc[df_pd['cluster'] == parent_1, 2] = parent_0

        if N - counter == k:
            for i in range(0, len(df_pd)):
                if df_pd.iloc[i, 2] == 'Non visited':
                    df_pd.iloc[i, 2] = i

            return df_pd


cl = get_mst(3, 450, sorted_edges, sorted_dist)


# Plot results
sns.lmplot(x="x1", y="x2", data=cl, fit_reg=False, hue='cluster', legend = True)
plt.show()