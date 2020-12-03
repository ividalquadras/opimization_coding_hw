import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import DistanceMetric
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Simulate dataframe
#x1 = np.random.uniform(low=0, high=2500, size=(200))
#x2 = np.random.uniform(low=5, high=3000, size=(200))

#df_pd = pd.DataFrame({'x1': x1, 'x2': x2}, columns=['x1', 'x2'])


def closest(x, clusters):
    c0 = []
    d0 = np.Inf

    for c in clusters:
        d = np.linalg.norm(c-x)
        if d < d0:
            c0 = c
            d0 = d
    return c0

def equal_clusters(c1, c2):
    for c in range(len(c1)):
        if np.linalg.norm(c1[c] - c2[c]) > 0.000000001:
            return False

    return True


def k_means(df_pd, k):

    clusters = df_pd.sample(k)['point'].to_list()

    df_pd['cluster'] = df_pd.apply(lambda x: closest(x.point, clusters), axis=1)

    initial_clustering = np.array(df_pd['point'])
    final_clustering = np.array(df_pd['cluster'])


    while not equal_clusters(final_clustering, initial_clustering):
        initial_clustering = final_clustering
        # Calcutate Centroids
        df_pd['tuple'] = df_pd['cluster'].map(tuple)

        def f(x):
            return x.mean()

        clusters = [np.round(v, 3) for v in df_pd.groupby('tuple')['point'].apply(lambda x: x.mean()).values]

        print(df_pd.groupby('tuple')['point'].count())

        # Assing points to Centroids
        df_pd['cluster'] = df_pd.apply(lambda x: closest(x.point, clusters), axis=1)

        final_clustering = np.array(df_pd['cluster'])

    return df_pd


if __name__ == '__main__':
    k_num = 15
    #seed = 8
#    class_data = make_blobs(n_samples=[100, 200, 150], n_features=2, random_state=seed)
#    df_aux = pd.DataFrame()
#    df_aux = pd.DataFrame(class_data[0])

    df_aux = pd.read_csv('/Users/ividalquadras/Desktop/synthetic_clean.csv')

    df_pd = pd.DataFrame()
    df_pd['point'] = df_aux.apply(np.array, axis=1)

    cl = k_means(df_pd, k_num)

    pl = pd.DataFrame({'x1': [v[0] for v in cl['point']], 'x2': [v[1] for v in cl['point']],
                  'cluster': [str(i) for i in cl['tuple']]})
    sns.lmplot( x="x1", y="x2", data=pl, fit_reg=False, hue='cluster', legend=False)
    plt.show()
def DB_index(df):
    df['tuple'] = df['cluster'].map(tuple)
    clusters = df.groupby(df["tuple"])
    dispersions = []
    separations = []
    for i in clusters.groups.keys():
        dis = []
        for m in df["point"][clusters.groups[i]]:
            dis.append(np.linalg.norm(np.array(i)-np.array(m)))
        dispersions.append((sum(dis)/len(dis))**0.5)
        seps = []
        for r in clusters.groups.keys():
            seps.append(np.linalg.norm(np.array(i)-np.array(r)))
        separations.append(seps)
    Ds = []
    for e in range(len(dispersions)):
        SoI = separations[e]
        Rs = []
        for t in range(len(SoI)):
            if SoI[t] != 0:
                Rs.append((dispersions[e]+dispersions[t]/SoI[t]))
        Ds.append(max(Rs))
    return np.average(Ds)
from scipy.spatial import distance
def Dunn_index(df):
    df['tuple'] = df['cluster'].map(tuple)
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