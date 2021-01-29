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
from scipy.spatial import distance as d


from sklearn.preprocessing import normalize
k1=normalize(kmeans_distances,norm='l2',axis=1,copy=False)
k2=normalize(kmeansplus_distances,norm='l2',axis=1,copy=False)
k3=normalize(cmeans_distances,norm='l2',axis=1,copy=False)

def index_maxdist(k,l):
    s=round(k*0.05)
    t=[]
    m=[]
    for i in range(s):
        t.append(l.index(max(l)))
        m.append(max(l))
        l.remove(max(l))
    return t,m

def new_clusters(k1,df_clustering,labels1):
    kd1=k1[:,0]
    kdist1=[]
    kdist2=[]
    kdist3=[]
    kvalue1=[]
    kvalue2=[]
    kvalue3=[]
    ind1=[]
    ind2=[]
    ind3=[]
    for i in range(569):
        if(labels1[i]==0):
            kdist1.append(kd1[i])
            ind1.append(i)
        if(labels1[i]==1):
            kdist2.append(kd1[i])
            ind2.append(i)
        if(labels1[i]==2):
            kdist3.append(kd1[i])
            ind3.append(i)
          
    for j in ind1:
        kvalue1.append(df_clustering.iloc[j])
    for j in ind2:
        kvalue2.append(df_clustering.iloc[j])
    for j in ind3:
        kvalue3.append(df_clustering.iloc[j])
    
    kvalue1=np.array(kvalue1)
    kvalue2=np.array(kvalue2)
    kvalue3=np.array(kvalue3)



    kmaxdis1,kmdis1=index_maxdist(len(kdist1),kdist1)
    kmaxdis2,kmdis2=index_maxdist(len(kdist2),kdist2)
    kmaxdis3,kmdis3=index_maxdist(len(kdist3),kdist3)
    
    
    kmaxdisvalues1=[]
    kmaxdisvalues2=[]
    kmaxdisvalues3=[]
    
    for i in kmaxdis1:
        kmaxdisvalues1.append(kvalue1[i])   
    for i in kmaxdis2:
        kmaxdisvalues2.append(kvalue2[i])
    for i in kmaxdis3:
        kmaxdisvalues3.append(kvalue3[i])   
    
    for i in kmaxdis1:
        kvalue1=np.delete(kvalue1,obj=i,axis=0)
    for i in kmaxdis2:
        kvalue2=np.delete(kvalue2,obj=i,axis=0)
    for i in kmaxdis3:
        kvalue3=np.delete(kvalue3,obj=i,axis=0)
    
    kmaxdisvalues1=np.array(kmaxdisvalues1)
    kmaxdisvalues2=np.array(kmaxdisvalues2)
    kmaxdisvalues3=np.array(kmaxdisvalues3)
    
    return kvalue1,kvalue2,kvalue3,kmaxdisvalues1,kmaxdisvalues2,kmaxdisvalues3,kmdis1,kmdis2,kmdis3,kmaxdis1,kmaxdis2,kmaxdis3

#for new kmeans
kvalue1,kvalue2,kvalue3,kmaxdisvalues1,kmaxdisvalues2,kmaxdisvalues3,kmdis1,kmdis2,kmdis3,kmaxdis1,kmaxdis2,kmaxdis3=new_clusters(k1,df_clustering,labels1)
"""
kmeansnew1 = KMeans(n_clusters=1,init='random',random_state=0).fit(kvalue1)
kmeans_centersnew=kmeansnew1.cluster_centers_
"""
kmeansnew2 = KMeans(n_clusters=1,init='random',random_state=0).fit(kvalue2)
kmeans_centersnew=  kmeansnew2.cluster_centers_

kmeansnew3 = KMeans(n_clusters=1,init='random',random_state=0).fit(kvalue3)
kmeans_centersnew= np.vstack((kmeans_centersnew, kmeansnew3.cluster_centers_))

kmeansnew = KMeans(n_clusters=2,init=kmeans_centersnew,n_init=1,random_state=0).fit(df_clustering)
kmeans_distancesnew = kmeansnew.fit_transform(df_clustering.values)
l1=kmeansnew.predict(df_clustering)
#---------------------------------------------------------------------------------

#for new cmeans
cvalue1,cvalue2,cvalue3,cmaxdisvalues1,cmaxdisvalues2,cmaxdisvalues3,cmdis1,cmdis2,cmdis3,cmaxdis1,cmaxdis2,cmaxdis3=new_clusters(k2,df_clustering,labels2)
"""
fcm1 = FCM(n_clusters=1)
fcm1.fit(cvalue1)
cmeans_centersnew = fcm1.centers
"""
fcm2 = FCM(n_clusters=1)
fcm2.fit(cvalue2)
cmeans_centersnew=  fcm2.centers

fcm3 = FCM(n_clusters=1)
fcm3.fit(cvalue3)
cmeans_centersnew= np.vstack((cmeans_centersnew, fcm3.centers))

cmeansnew = KMeans(n_clusters=2,init=cmeans_centersnew,n_init=1,random_state=0).fit(df_clustering)
cmeans_distancesnew = cmeansnew.fit_transform(df_clustering.values)
l2=cmeansnew.predict(df_clustering)
#---------------------------------------------------------------------------------

#for new kmeans++
kpvalue1,kpvalue2,kpvalue3,kpmaxdisvalues1,kpmaxdisvalues2,kpmaxdisvalues3,kpmdis1,kpmdis2,kpmdis3,kpmaxdis1,kpmaxdis2,kpmaxdis3=new_clusters(k3,df_clustering,labels3)
"""
kmeansplusnew1 = KMeans(n_clusters=1,init='k-means++',random_state=0).fit(kpvalue1)
kmeansplus_centersnew=kmeansplusnew1.cluster_centers_
"""
kmeansplusnew2 = KMeans(n_clusters=1,init='k-means++',random_state=0).fit(kpvalue2)
kmeansplus_centersnew= kmeansplusnew2.cluster_centers_

kmeansplusnew3 = KMeans(n_clusters=1,init='k-means++',random_state=0).fit(kpvalue3)
kmeansplus_centersnew= np.vstack((kmeansplus_centersnew, kmeansplusnew3.cluster_centers_))

kmeansplusnew = KMeans(n_clusters=2,init=kmeansplus_centersnew,n_init=1,random_state=0).fit(df_clustering)
kmeansplus_distancesnew = kmeansplusnew.fit_transform(df_clustering.values)
l3=kmeansplusnew.predict(df_clustering)

kn1=normalize(kmeans_distancesnew,norm='l2',axis=1,copy=False)
kn2=normalize(cmeans_distancesnew,norm='l2',axis=1,copy=False)
kn3=normalize(kmeansplus_distancesnew,norm='l2',axis=1,copy=False)
#---------------------------------------------------------------------------------




def selecting_newclusters(maxdisvalues,mdis,maxdis):
    t=len(maxdisvalues)-1
    s=[]
    m=[]
    if(t>0):
        for i in range (t):
            a1=d.euclidean(maxdisvalues[t],maxdisvalues[i])
            s.append(a1)
    for j in range(t):
        if(s[j]<0.25*mdis[t]):
            m.append(maxdis[j])
    return m,s

#for kmeans
kc1,kcdis1=selecting_newclusters(kmaxdisvalues1,kmdis1,kmaxdis1)   
kc2,kcdis2=selecting_newclusters(kmaxdisvalues2,kmdis2,kmaxdis2)
kc3,kcdis3=selecting_newclusters(kmaxdisvalues3,kmdis3,kmaxdis3)
#for cmeans
cc1,ccdis1=selecting_newclusters(cmaxdisvalues1,cmdis1,cmaxdis1)   
cc2,ccdis2=selecting_newclusters(cmaxdisvalues2,cmdis2,cmaxdis2)
cc3,ccdis3=selecting_newclusters(cmaxdisvalues3,cmdis3,cmaxdis3)
#for kmeans++
kpc1,kpcdis1=selecting_newclusters(kpmaxdisvalues1,kpmdis1,kpmaxdis1)   
kpc2,kpcdis2=selecting_newclusters(kpmaxdisvalues2,kpmdis2,kpmaxdis2)
kpc3,kpcdis3=selecting_newclusters(kpmaxdisvalues3,kpmdis3,kpmaxdis3)
#----------------------------------------------------------------------



#function to create hyperbola matrix 
def hyperbola(distance):
    s=[]
    for i in range(569):          
        a =[] 
        for j in range(2):      
             a.append(1/distance[i][j]) 
        s.append(a)
    s=np.array(s)
    return s

#function for probability finding for each clustering method
def probabilities(k,sum):
    s=[]
    for i in range(569):          
        a =[] 
        for j in range(2):      
             a.append(k[i][j]/sum[i]) 
        s.append(a)
    s=np.array(s)
    return s

kmeans_hyperbola=[]
kmeans_hyperbola=hyperbola(kn1)
sum1=kmeans_hyperbola.sum(axis=1)

kmeansplus_hyperbola=[]
kmeansplus_hyperbola=hyperbola(kn2)
sum2=kmeansplus_hyperbola.sum(axis=1)
   
cmeans_hyperbola=[]
cmeans_hyperbola=hyperbola(kn3)
sum3=cmeans_hyperbola.sum(axis=1)

kmeans_probabilities=[]
kmeans_probabilities=probabilities(kmeans_hyperbola,sum1)
#s1=kmeans_probabilities.sum(axis=1)

kmeansplus_probabilities=[]
kmeansplus_probabilities=probabilities(kmeansplus_hyperbola,sum2)
#after relabelling
#s2=kmeansplus_probabilities.sum(axis=1)

cmeans_probabilities=[]
cmeans_probabilities=probabilities(cmeans_hyperbola,sum3)
#s3=cmeans_probabilities.sum(axis=1)


#calculate euclidean distances for each centers
from scipy.spatial import distance
a=[]
for i in range(0,2):
    for j in range(0,2):
        a.append(distance.euclidean(kmeansplus_centersnew[j], kmeans_centersnew[i]))
a  = np.reshape(a, (-1, 2))
        
b=[]
for i in range(0,2):
    for j in range(0,2):
        b.append(distance.euclidean(cmeans_centersnew[j], kmeans_centersnew[i]))
b=np.reshape(b, (-1, 2))


silk=(silhouette_score(df_clustering, labels1)) 
silc=(silhouette_score(df_clustering, labels2)) 
silkplus=(silhouette_score(df_clustering, labels3)) 
sumsil=silk+silc+silkplus
#calculate weighted product
def weighted_products(k1,k2,k3):
    wproducts=[]
    for i in range(569):
        l=[]
        for j in range(2):
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
        if(p[i]<1/2+0.08):
            outlier.append(i)
        else:
            continue
    return outlier

outlier=outlier_checking(max_wproducts)

#deleting the outliers from the data
df_clustering_outlier=df_clustering.drop(outlier, axis = 0)



#for checking the quality of clustering 




labels_outlier1 = kmeansnew.predict(df_clustering_outlier)
labels_outlier2 = cmeansnew.predict(df_clustering_outlier)
labels_outlier3 = kmeansplusnew.predict(df_clustering_outlier)

print("\n\tDB SCORE (LESS IS BETTER)\n")
print("BEFORE OUTLIER DELETION")  
print("Kmeans: ",davies_bouldin_score(df_clustering, l1)) 
print("Cmeans: ",davies_bouldin_score(df_clustering, l2)) 
print("Kmeans++: ",davies_bouldin_score(df_clustering, l3)) 
print()
print("AFTER OUTLIER DELETION") 
print("Kmeans: ",davies_bouldin_score(df_clustering_outlier, labels_outlier1)) 
print("Cmeans: ",davies_bouldin_score(df_clustering_outlier, labels_outlier2)) 
print("Kmeans++: ",davies_bouldin_score(df_clustering_outlier, labels_outlier3)) 

print("\n\tSilhouette Index  (MORE IS BETTER)\n")
print("BEFORE OUTLIER DELETION")  
print("Kmeans: ",silhouette_score(df_clustering, l1)) 
print("Cmeans: ",silhouette_score(df_clustering, l2)) 
print("Kmeans++: ",silhouette_score(df_clustering, l3)) 
print()
print("AFTER OUTLIER DELETION") 
print("Kmeans: ",silhouette_score(df_clustering_outlier, labels_outlier1)) 
print("Cmeans: ",silhouette_score(df_clustering_outlier, labels_outlier2)) 
print("Kmeans++: ",silhouette_score(df_clustering_outlier, labels_outlier3)) 

print("\n\tCalinski-Harabasz Index  (MORE IS BETTER)\n")
print("BEFORE OUTLIER DELETION")  
print("Kmeans: ",calinski_harabasz_score(df_clustering, l1)) 
print("Cmeans: ",calinski_harabasz_score(df_clustering, l2)) 
print("Kmeans++: ",calinski_harabasz_score(df_clustering, l3)) 
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

data.to_csv('newalg_wisconsin_outlier.csv') 

