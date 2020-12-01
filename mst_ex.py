import numpy as np
from numpy import random
import pandas as pd
from sklearn.neighbors import DistanceMetric

# Simulate dataframe
x1 = np.random.uniform(low=0, high=2500, size=(200))
x2 = np.random.uniform(low=5, high=3000, size=(200))

df = pd.DataFrame({'x1': x1, 'x2': x2}, columns=['x1', 'x2'])

# Compute distances
dist = DistanceMetric.get_metric('euclidean')  # euclidean distance
df = df.to_numpy()

distance = dist.pairwise(df)  # matrix with distances

# Create matrix for sorted origin and end point of the edge
n = distance.shape[1]
point_origin_and_end = list(zip(np.argsort(distance, axis=None).__divmod__(n)))
point_origin_and_end = list(zip(point_origin_and_end[0][0], point_origin_and_end[1][0]))

# Sorted distances
sorted_dist = np.argsort(distance, axis=None)
sorted_dist = sorted_dist[0::2]

# Recover the edce (origin index and end index of the edge) and sort them
sorted_edges = np.transpose(
    np.unravel_index(sorted_dist, distance.shape))  # Array with origin and end vertex number of the sorted edges

# Convert them to lists to work with them and drop the first 200 distances because they are for the same points
# There are 39800 edges (200*200-200)
sorted_dist = sorted_dist.tolist()[200:]
sorted_edges = sorted_edges.tolist()[200:]

# Create a list with all the vertices to keep track if added in the MSP
vertices = [[i, False] for i in range(0, 200)]


# Start the algorithm

def get_mst(k, N, sorted_edges, sorted_dist):
    clusters = {1: sorted_edges[0]}  # dictionary with clusters and vertices in each cluster
    mst_cost = sorted_dist[0]  # cost of the MST
    counter = 1  # counter with the number of edges added into the MST

    # set vertex 0 as of first edge visited
    vertices[sorted_edges[0][0]] = [sorted_edges[0][0], True]

    # set vertex 1 as of first edge visited
    vertices[sorted_edges[0][1]] = [sorted_edges[0][1], True]

    for edge in range(1, len(sorted_edges)):

        ######---------------------- Look if vertex belong to a cluster ----------------------######

        parent_0 = None  # by default assume the vertex does not belong to any Cluster
        parent_1 = None  # by default assume the vertex does not belong to any Cluster

        # check if vertex 0 actually belongs to some cluster (parent)
        for cluster in clusters:
            if sorted_edges[edge][0] in clusters[cluster]:
                parent_0 = cluster
                break

        # check if vertex 1 actually belongs to some cluster (parent)
        for cluster in clusters:
            if sorted_edges[edge][1] in clusters[cluster]:
                parent_1 = cluster
                break

        ######------------------------------- Start the merging -------------------------------######

        # if the vertices are in the same cluster, do nothing bc I would create a cycle
        if (parent_0 == parent_1) and (parent_0 is not None):
            pass

            # if the vertices are not in any cluster, then create one with the 2 vertices
        elif (parent_0 is None) and (parent_1 is None):
            # add a cluster with both vertices
            clusters[max(clusters) + 1] = [sorted_edges[edge][0], sorted_edges[edge][1]]

            # set vertex 0 as visited
            vertices[sorted_edges[edge][0]] = [sorted_edges[edge][0], True]

            # set vertex 1 as visited
            vertices[sorted_edges[edge][1]] = [sorted_edges[edge][1], True]

            # add the cost of the edge to the cost of the MST
            mst_cost += sorted_dist[edge]

            # increased the counter by one as we have added one edge to the MST
            counter += 1

            # if one of the vertices is not in any cluster, then add this one to the cluster of the other vertex
        elif (parent_0 is None) or (parent_1 is None):

            counter += 1
            mst_cost += sorted_dist[edge]

            if parent_0 is None:
                # add vertex 0 to the cluster of vertex 1
                clusters[parent_1] = clusters[parent_1] + [sorted_edges[edge][0]]

                # set vertex 0 as visited
                vertices[sorted_edges[edge][0]] = [sorted_edges[edge][0], True]

            else:
                # add vertex 1 to the cluster of vertex 0
                clusters[parent_0] = clusters[parent_0] + [sorted_edges[edge][1]]

                # set vertex 1 as visited
                vertices[sorted_edges[edge][1]] = [sorted_edges[edge][1], True]

                # if both belong to one cluster (but different ones)
        else:
            # add the vertices of the cluster of vertex 2 to the cluster of vertrex 1
            clusters[parent_0] = clusters[parent_0] + clusters[parent_1]

            # delete cluster of vertex 2 from the dictionary of clusters
            del clusters[parent_1]

            mst_cost += sorted_dist[edge]
            counter += 1

        # Finally stop the merging if we have already K clusters
        if N - counter == k:
            break

    ######----------------- Add to the cluster's dictionary the non-visited vertices ----------------######
    for i in vertices:
        if i[1] == False:
            clusters[max(clusters) + 1] = i[0]

    print(clusters)


get_mst(10, 200, sorted_edges, sorted_dist)
