import numpy as np
import pandas as pd
from sklearn.neighbors import DistanceMetric
from scipy.spatial import distance
import time


def mst(data_path, k):

    df_pd = pd.read_csv(data_path)  # import the data
    N = len(df_pd)  # length of the data
    df_pd.columns = [i for i in range(0, len(df_pd.columns))]  # name the columns with numbers

    # Compute distances
    dist = DistanceMetric.get_metric('euclidean')
    df = df_pd.to_numpy()

    distance = dist.pairwise(df)  # matrix with the pair-wise euclidean distance between all data points

    # Sort the edges in ascending order (according to the distance in between vertices)
    sorted_edges = np.transpose(np.unravel_index(np.argsort(distance, axis=None), distance.shape)).tolist()

    # Remove the first N zeros (distance between the same data points) and as they are duplicated, choose one for each 2
    sorted_edges = sorted_edges[N::2]

    # Add a cluster column to keep track of the point's cluster, if it does not belong to any, it will be 'Non visited'
    df_pd['cluster'] = ['Non visited' for i in range(0, N)]

    # Start algorithm

    # Initiate a dictionary with cluster and the data points it has
    cluster_dic = {-1: []}

    # Initiate a counter for the number of edges we add
    counter = 0
    t0 = time.time()
    for edge in sorted_edges:  # iterate over all the edges (sorted)

        parent_0 = df_pd.loc[edge[0], 'cluster']  # set the cluster to which the first data point of the edge belongs to
        parent_1 = df_pd.loc[
            edge[1], 'cluster']  # set the cluster to which the second data point of the edge belongs to

        # If both data points belong to the same cluster, then do nothing because it would create a cycle
        if (parent_0 == parent_1) and (parent_0 != 'Non visited'):
            pass

        # If both data points have no cluster assigned, create a new cluster with both data points
        elif (parent_0 == 'Non visited') and (parent_1 == 'Non visited'):
            counter += 1
            max_cluster = max(cluster_dic)
            cluster_dic[max_cluster + 1] = [edge[0]]
            cluster_dic[max_cluster + 1] += [edge[1]]

            # Keep track that these two data points have now a cluster by including the cluster number to df_pd
            df_pd.loc[edge[0], 'cluster'] = max_cluster + 1
            df_pd.loc[edge[1], 'cluster'] = max_cluster + 1

        # If some of the data points does not belong to any cluster, add the other data point of the edge to the cluster
        # of the first one and keep track that this data point has now a cluster by including the cluster to df_pd
        elif (parent_0 == 'Non visited') or (parent_1 == 'Non visited'):
            counter += 1
            if parent_0 == 'Non visited':
                cluster_dic[parent_1] += [edge[0]]
                df_pd.loc[edge[0], 'cluster'] = parent_1
            else:
                cluster_dic[parent_0] += [edge[1]]
                df_pd.loc[edge[1], 'cluster'] = parent_0

        # If the two data points belong to different cluster, add the vertices of the second cluster to the first,
        # delete the second cluster and change the cluster in df_pd for points in second cluster
        else:
            counter += 1
            cluster_dic[parent_0] += cluster_dic[parent_1]
            del cluster_dic[parent_1]

            df_pd.loc[df_pd['cluster'] == parent_1, 'cluster'] = parent_0

        # Stop iterating once we have N clusters, that is, we have added K == N - edges we have added and add these
        # points as 'alone' clusters (clusters with only one data point)
        if N - counter == k:
            for i in range(0, N):
                if df_pd.loc[i, 'cluster'] == 'Non visited':
                    df_pd.loc[i, 'cluster'] = max(cluster_dic) + 1
                    cluster_dic[max(cluster_dic)] = i
            t1 = time.time()
            print(t1-t0)
            return df_pd
