#Jonathan Gee
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc
from timeit import default_timer as timer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
import operator


#common category tags to ignore
ignore = ["Restaurants", "Food", "Nightlife"]

#prints the top 5 categories for each cluster
#grouped is an dataframe with "clusters" from 0 to n and "categories"
#n is the number of clusters
def print_Categories(grouped, n):

	cat_list = [dict() for x in range(n)]

	for index, row in grouped.iterrows():
		for x in range(n):
			if (row["clusters"] == x):
				cats = row["categories"]
				words = [x.strip() for x in cats.split(",")]
				for word in words:
					if word not in ignore:
						if word in cat_list[x]:
							cat_list[x][word] += 1
						else:
							cat_list[x][word] = 1
				break


	for x in range(n):
		print("\n\n", x , "category dictionary")
		sorted_x = sorted(cat_list[x].items(), key=lambda kv: kv[1])
		sorted_x = list(reversed(sorted_x))
		for y in range(5):
			print(sorted_x[y])
	return


#prints the count of unique items in array label
def label_count(labels):

	lab_count = dict()

	for x in labels:
		if x in lab_count:
			lab_count[x] += 1
		else:
			lab_count[x] = 1

	print(lab_count)

	return


#load data
data = pd.read_csv("cleaned_merged.csv")
print("data size:", data.shape)

#randomize order of data
data = data.sample(frac = 1, random_state = 1)

#separate into train and test data
#train = data.iloc[0:35500,:]

#use all data for agg, DBScan, Birch
train = data
test = data.iloc[35501:,:]


#take small subset for algorithms
#data = data.sample(n = 25000, random_state = 0)

#separate the checkin data to cluster on
checkins = train.iloc[:, 12:]
checkins = checkins.fillna(0)
#print(checkins.head())

#for predicting a new value with given time
# predict = checkins.iloc[0,:]
# predict = predict * 0
# predict["time.Fri-18"] = 1
# predict = predict.values
# predict = predict.reshape(1,-1)

#number of clusters to use
n = 5

print("loaded data")
start = timer()

print("clustering data...")

#-------------------------------------------------------------------------------------------

cluster = KMeans(n_clusters = n, random_state = 0).fit(checkins)
	#works, but has a pretty bad split {0: 40954, 3: 2868, 4: 422, 2: 57, 1: 17}
#cluster = MiniBatchKMeans(n_clusters = n, random_state = 0, batch_size = 500, max_iter = 200).fit(checkins)
	#probably the best one good spread {0: 8379, 2: 26735, 4: 4450, 1: 3669, 3: 1085}

#test these methods

#cluster = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward').fit(checkins)
	#agg works with 25000, 92 sec, but pretty bad results, {2: 22491, 0: 2304, 4: 170, 3: 29, 1: 6}
#cluster = DBSCAN(eps = 5, min_samples = 10).fit(checkins)
	#extremely slow for 25000, 173 sec, only outputs 2 categories {-1: 16778, 0: 8222}
#cluster = Birch(n_clusters = n).fit(checkins)
	#Birch works with 25000, 100 sec, but uses agg clustering?
#cluster = SpectralClustering(n_clusters = n, random_state = 0, affinity = "nearest_neighbors").fit(checkins)
	#doesnt work?

#-------------------------------------------------------------------------------------------

labels = cluster.labels_

end = timer()

print("elapsed time: " , end - start , " seconds")

#display the clusters
print("\n\nClusters:")
label_count(labels)

#attach the clusters to other data
train["clusters"] = labels
info = ["categories", "name", "clusters"]
grouped = train[info]
grouped = grouped.fillna("")

#display categories within clusters
print_Categories(grouped, n)


#test wont work for some of the clustering methods since they dont support predict
print("\n\n---------    training data categories    ---------\n\n")

test_checkins = test.iloc[:, 12:]
test_checkins = test_checkins.fillna(0)
predict_labels = cluster.predict(test_checkins)
test["clusters"] = predict_labels
print("\n\nClusters:")
label_count(predict_labels)

grouped = test[info]
grouped = grouped.fillna("")

print_Categories(grouped, n)







