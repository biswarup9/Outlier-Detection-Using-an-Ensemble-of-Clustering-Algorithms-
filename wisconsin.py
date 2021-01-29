# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:15:42 2020

@author: Biswarup Ray
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import davies_bouldin_score,silhouette_score,calinski_harabasz_score
import copy

df= pd.read_csv("data.csv")

df.drop(["Unnamed: 32", "id"], axis = 1, inplace = True)

#dropping the labels from the data
df_clustering = df.drop(["diagnosis"], axis = 1)
data=copy.deepcopy(df)

#data without diagnosis label
plt.figure(figsize = (10, 10))
plt.scatter(df_clustering["radius_mean"], df_clustering["texture_mean"])
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.show()

#------------------------------------------------------------
#for Kmeans algo
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,init='random',random_state=0).fit(df_clustering)
kmeans_centers=kmeans.cluster_centers_
#the euclidian distance of each point from each cluster centres            
kmeans_distances = kmeans.fit_transform(df_clustering.values)
labels1 = kmeans.predict(df_clustering)




#-------------------------------------------------------------
#for Cmeans algo

from fcmeans import FCM
fcm = FCM(n_clusters=3)
fcm.fit(df_clustering.values)
cmeans_centers = fcm.centers
#after relabelling
cmeans_centers[[0,1]]=cmeans_centers[[1,0]]
cmeans_centers[[1,2]]=cmeans_centers[[2,1]]
cmeans = KMeans(n_clusters=3,init=cmeans_centers,n_init=1,random_state=0).fit(df_clustering)
cmeans_distances = cmeans.fit_transform(df_clustering.values)
labels2 = cmeans.predict(df_clustering)




#-------------------------------------------------------------
#for kmeans++
# function to compute euclidean distance 
kmeansplus1 = KMeans(n_clusters=3,init='k-means++',n_init=1,random_state=0).fit(df_clustering)
kmeansplus1_centers=kmeansplus1.cluster_centers_
kmeansplus1_centers[[1,2]]=kmeansplus1_centers[[2,1]]
kmeansplus1_centers[[0,1]]=kmeansplus1_centers[[1,0]]
#after relabelling
kmeansplus = KMeans(n_clusters=3,init=kmeansplus1_centers,n_init=1,random_state=0).fit(df_clustering)
kmeansplus_distances = kmeansplus.fit_transform(df_clustering.values)
kmeansplus_centers=kmeansplus.cluster_centers_
labels3 = kmeansplus.predict(df_clustering)



#-------------------------------------------------------------






from sklearn.preprocessing import normalize
k1=normalize(kmeans_distances,norm='l2',axis=1,copy=False)
k2=normalize(kmeansplus_distances,norm='l2',axis=1,copy=False)
k3=normalize(cmeans_distances,norm='l2',axis=1,copy=False)

#function to create exponent matrix 
def exponent(distance):
    s=[]
    for i in range(569):          
        a =[] 
        for j in range(3):      
             a.append(1/distance[i][j]) 
        s.append(a)
    s=np.array(s)
    return s

#function for probability finding for each clustering method
def probabilities(k,sum):
    s=[]
    for i in range(569):          
        a =[] 
        for j in range(3):      
             a.append(k[i][j]/sum[i]) 
        s.append(a)
    s=np.array(s)
    return s

kmeans_exponent=[]
kmeans_exponent=exponent(k1)
sum1=kmeans_exponent.sum(axis=1)

kmeansplus_exponent=[]
kmeansplus_exponent=exponent(k2)
sum2=kmeansplus_exponent.sum(axis=1)
   
cmeans_exponent=[]
cmeans_exponent=exponent(k3)
sum3=cmeans_exponent.sum(axis=1)

kmeans_probabilities=[]
kmeans_probabilities=probabilities(kmeans_exponent,sum1)
#s1=kmeans_probabilities.sum(axis=1)

kmeansplus_probabilities=[]
kmeansplus_probabilities=probabilities(kmeansplus_exponent,sum2)
#after relabelling
#s2=kmeansplus_probabilities.sum(axis=1)

cmeans_probabilities=[]
cmeans_probabilities=probabilities(cmeans_exponent,sum3)
#s3=cmeans_probabilities.sum(axis=1)


#calculate euclidean distances for each centers
from scipy.spatial import distance
a=[]
for i in range(0,3):
    for j in range(0,3):
        a.append(distance.euclidean(kmeansplus_centers[j], kmeans_centers[i]))
a  = np.reshape(a, (-1, 3))
        
b=[]
for i in range(0,3):
    for j in range(0,3):
        b.append(distance.euclidean(cmeans_centers[j], kmeans_centers[i]))
b=np.reshape(b, (-1, 3))


silk=(silhouette_score(df_clustering, labels1)) 
silc=(silhouette_score(df_clustering, labels2)) 
silkplus=(silhouette_score(df_clustering, labels3)) 
sumsil=silk+silc+silkplus
#calculate weighted product
def weighted_products(k1,k2,k3):
    wproducts=[]
    for i in range(569):
        l=[]
        for j in range(3):
            l.append((k1[i][j]**(silk/sumsil))*(k2[i][j]**(silc/sumsil))*(k3[i][j]**(silkplus/sumsil)))
        wproducts.append(l)  
    wproducts=np.array(wproducts)     
    return wproducts

wproducts=weighted_products(kmeans_probabilities,cmeans_probabilities,kmeansplus_probabilities)
max_wproducts = np.amax(wproducts, 1)

#outlier checking condition
def outlier_checking(p):
    outlier=[]
    for i in range(569):
        if(p[i]<1/3+0.10):
            outlier.append(i)
        else:
            continue
    return outlier

outlier=outlier_checking(max_wproducts)

#deleting the outliers from the data
df_clustering_outlier=df_clustering.drop(outlier, axis = 0)



#for checking the quality of clustering 


#DUNN INDEX
def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000
    
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])
            
    return np.min(values)
    
def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])
    
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])
            
    return np.max(values)

def dunn(k_list):
    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])
        
        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di



labels_outlier1 = kmeans.predict(df_clustering_outlier)
labels_outlier2 = cmeans.predict(df_clustering_outlier)
labels_outlier3 = kmeansplus.predict(df_clustering_outlier)

print("\n\tDB SCORE (LESS IS BETTER)\n")
print("BEFORE OUTLIER DELETION")  
print("Kmeans: ",silhouette_score(df_clustering, labels1)) 
print("Cmeans: ",silhouette_score(df_clustering, labels2)) 
print("Kmeans++: ",silhouette_score(df_clustering, labels3)) 
print()
print("AFTER OUTLIER DELETION") 
print("Kmeans: ",davies_bouldin_score(df_clustering_outlier, labels_outlier1)) 
print("Cmeans: ",davies_bouldin_score(df_clustering_outlier, labels_outlier2)) 
print("Kmeans++: ",davies_bouldin_score(df_clustering_outlier, labels_outlier3)) 

print("\n\tSilhouette Index  (MORE IS BETTER)\n")
print("BEFORE OUTLIER DELETION")  
print("Kmeans: ",silhouette_score(df_clustering, labels1)) 
print("Cmeans: ",silhouette_score(df_clustering, labels2)) 
print("Kmeans++: ",silhouette_score(df_clustering, labels3)) 
print()
print("AFTER OUTLIER DELETION") 
print("Kmeans: ",silhouette_score(df_clustering_outlier, labels_outlier1)) 
print("Cmeans: ",silhouette_score(df_clustering_outlier, labels_outlier2)) 
print("Kmeans++: ",silhouette_score(df_clustering_outlier, labels_outlier3)) 

print("\n\tCalinski-Harabasz Index  (MORE IS BETTER)\n")
print("BEFORE OUTLIER DELETION")  
print("Kmeans: ",calinski_harabasz_score(df_clustering, labels1)) 
print("Cmeans: ",calinski_harabasz_score(df_clustering, labels2)) 
print("Kmeans++: ",calinski_harabasz_score(df_clustering, labels3)) 
print()
print("AFTER OUTLIER DELETION") 
print("Kmeans: ",calinski_harabasz_score(df_clustering_outlier, labels_outlier1)) 
print("Cmeans: ",calinski_harabasz_score(df_clustering_outlier, labels_outlier2)) 
print("Kmeans++: ",calinski_harabasz_score(df_clustering_outlier, labels_outlier3)) 

arr=[]
for i in range(len(data)):
    arr.append(0)
    
for j in range(len(outlier)):
    arr[outlier[j]]=1
    
data["outlier"]=arr

data.to_csv('wisconsin_outlier.csv') 