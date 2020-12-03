from mst_ex import mst
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from k_means_ex import k_means, db_k_means, dunn_index
import pandas as pd
import numpy as np
import time


#######-------------------------------------  Minimum Spanning Tree exercise -------------------------------------#######

# To get the data points with the assigned clusters, insert the path of the data file and the number of trees desired
# The Davies Bouldin will be automatically computed
# The plot will appear if the data is 2-dimensional
# For taking a look at the function refer to the mst_ex.py file

cl = mst('/Users/ividalquadras/Desktop/synthetic_clean.csv', 15)

# Data points with its assigned cluster
print('Data points with its assigned cluster')
print(cl)

# Davies Bouldin index score
db_index = sklearn.metrics.davies_bouldin_score(cl.drop(['cluster'], axis=1), cl['cluster'])
print('Davies Bouldin Index = ' + f'{db_index}')


# Plot results if data is 2 dimensional
if len(cl.columns) == 3:
    cl = cl.rename(columns = {0:'x1', 1:'x2'})
    sns.lmplot(x='x1', y='x2', data=cl, fit_reg=False, hue='cluster', legend=False)
    plt.show()



#######-------------------------------------  K-means exercise -------------------------------------#######

# To get the data points with the assigned clusters, insert the path of the data file and the number of trees desired
# The Davies Bouldin will be automatically computed
# The plot will appear if the data is 2-dimensional
# For taking a look at the function refer to the k_means_ex.py file

cl_k = k_means('/Users/ividalquadras/Desktop/synthetic_clean.csv', 15)

# Data points with its assigned cluster
print("Data points with its assigned cluster's centroids")
print(cl_k)


# Davies Bouldin index score
print('Davies Bouldin Index = ' + f'{db_k_means(cl_k)}')

# Dunn index score
print('Dunn Index = ' + f'{dunn_index(cl_k)}')


# Plot results if data is 2 dimensional
db = pd.DataFrame(cl_k['point'].to_list(), columns=[i for i in range(0, len(cl_k['point'].to_list()[0]))])
db['cluster'] = cl_k['cluster']
if len(db.columns) == 3:
    db = db.rename(columns = {0:'x1', 1:'x2'})
    db['cluster'] = db['cluster'].apply(lambda x: str(x))
    sns.lmplot(x='x1', y='x2', data=db, fit_reg=False, hue='cluster', legend=False)
    plt.show()
