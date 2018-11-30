import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.cluster import AgglomerativeClustering
import numpy as np  
import scipy.cluster.hierarchy as shc
#%matplotlib inline
'exec(%matplotlib inline)'

ClusterDF = pd.read_csv("checkincount_time_byid.csv")

arr = np.array(ClusterDF[['time','check_in_count']])
indices = np.random.randint(0,arr.shape[0],200)
X = arr[indices]

plt.figure(figsize=(9, 10))  
plt.title("Data Dendrogram Single Link")  
dend = shc.dendrogram(shc.linkage(X, method='ward'))  
print ('\nprint dend single link\n')
print(dend)

plt.show() 



cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
print(cluster.fit(X))

plt.figure(figsize=(9, 10))  
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')  
plt.title("Single Link")
plt.show()


#Average Link
plt.figure(figsize=(9, 10))  
plt.title("Data Dendrogram Average Link")  
dend = shc.dendrogram(shc.linkage(X, method='average'))  
print ('\nprint dend average link\n')
print(dend)

plt.show() 



cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')  
print(cluster.fit(X))

plt.figure(figsize=(9, 10))  
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')  
plt.title("Average Link")
plt.show() 


#Complete Link
plt.figure(figsize=(9, 10))  
plt.title("Data Dendrogram Complete Link")  
dend = shc.dendrogram(shc.linkage(X, method='complete'))  
print ('\nprint dend complete link\n')
print(dend)

plt.show() 



cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')  
print(cluster.fit(X))

plt.figure(figsize=(9, 10))  
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')  
plt.title("Complete Link")
plt.show() 

