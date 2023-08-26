#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


# In[2]:


# PCA algorithm 
import numpy as np
from scipy.linalg import svd 

def pca(dim,inp_matrix):
    length = len(inp_matrix)
    delta_hat= np.empty((length,0))
    identity_matrix = np.identity(length)
    e_vector = np.ones(length)
    delta_tilde= np.dot((identity_matrix-((np.multiply(e_vector,e_vector.reshape(length,1)))/length)),inp_matrix)
    u_leftVector,singular_val,v_rightVectorTranspose = svd(delta_tilde)
    for i in range(0,dim):
        arr= np.dot(singular_val[i],u_leftVector[:,i]).reshape(len(u_leftVector),1)
        delta_hat = np.append(delta_hat,arr,axis = 1)

    return delta_hat,u_leftVector,singular_val,v_rightVectorTranspose,delta_tilde


# In[3]:


import pandas as pd
import urllib.request
import tarfile
import io

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz"


response = urllib.request.urlopen(url)
tar_file = io.BytesIO(response.read())
tar = tarfile.open(fileobj=tar_file)
rna_seq_df= pd.read_csv(tar.extractfile("TCGA-PANCAN-HiSeq-801x20531/data.csv"))
label_df = pd.read_csv(tar.extractfile("TCGA-PANCAN-HiSeq-801x20531/labels.csv"))


# In[4]:


rna_merged_df= pd.merge(rna_seq_df, label_df, on='Unnamed: 0')
rna_merged_df = rna_merged_df.drop(columns=['Unnamed: 0'])


# ### 1.

# In[5]:


scaled_rna_data = StandardScaler().fit_transform(rna_merged_df.iloc[:,:-1].values)
rna_merged_df.iloc[:,:-1] = pd.DataFrame(scaled_rna_data, index=rna_merged_df.iloc[:,:-1].index, columns=rna_merged_df.iloc[:,:-1].columns)


# In[6]:


pca_matrix_2, u_leftVector, singular_val, v_rightVectorTranspose, delta_tilde = pca(2,np.array(rna_merged_df.iloc[:,:-1]))


# In[7]:


pca_matrix_2


# In[8]:


plt.scatter(x= pca_matrix_2[:,0], y= pca_matrix_2[:,1])
plt.savefig('scatterplot q1')
plt.clf()


# In[9]:


np.corrcoef(x=pca_matrix_2[:,0],y= pca_matrix_2[:,1])[0,1]


# In[10]:


## the correlation coefficient between PC1 and PC2 is 1.1783356349502421e-16 which is approximately around 0. 
# Hence PC1 and PC2 are not correlated.


# In[11]:


perct_variance = (singular_val ** 2) / np.sum(singular_val ** 2)
perct_variance


# In[12]:


total_var= sum(perct_variance[:2])*100
total_var


# In[13]:


## PC1 and PC2 explains about 19.29 percent of variance from the whole dataset.


# ## 2.

# In[14]:


perct_variance = (singular_val ** 2) / np.sum(singular_val ** 2)


# In[15]:


plt.plot(perct_variance)
plt.savefig('curve bend pca')
plt.clf()


# In[16]:


for i in range(0,len(perct_variance)):
    total_var1 = sum(perct_variance[:i])*100
    if (total_var1 >= 75):
        print(f"The number of components where percentage variance of total 75 percent is reached is {i}")
        break


# In[17]:


for i in range(1,len(perct_variance)):
    total_var1 = perct_variance[i]*100
    if(total_var1 <= 1):
        print(i)
        break


# ### 3.

# In[18]:


# loading
pca_all_data, u_leftVector, singular_val, v_rightVectorTranspose, delta_tilde = pca(len(rna_merged_df),np.array(rna_merged_df.iloc[:,:-1]))

n = len(rna_merged_df)-1

loadings_values = v_rightVectorTranspose[:10,:].T * np.sqrt(np.array((singular_val**2)/n)[:10])

load_mat = pd.DataFrame(loadings_values, index=list(rna_merged_df.columns).remove('Class'))


plt.boxplot(load_mat)
plt.savefig('loadings boxplot')
plt.show()


# In[19]:


perct_variance1 = np.cumsum(perct_variance)
plt.plot(perct_variance1)
plt.savefig('curve bend pca2')
plt.show()
plt.clf()


# ## 4
# 

# In[20]:


pca_all_data, u_leftVector, singular_val, v_rightVectorTranspose, delta_tilde = pca(len(rna_merged_df),np.array(rna_merged_df.iloc[:,:-1]))


# In[21]:


perct_variance = (singular_val ** 2) / np.sum(singular_val ** 2)


# In[22]:


for i in range(0,len(perct_variance)):
    total_var1 = sum(perct_variance[:i])*100
    if (total_var1 >= 90):
        print(f"The number of components where percentage variance of total 90 percent is reached is {i}")
        break


# In[23]:


pca_deltar= pca_all_data[:,:373]


# In[24]:


pca_deltar.shape


# In[25]:


rna_merged_df.Class.unique()


# In[26]:


#kmeans algorithm ck
import numpy as np
import random
import math

def eud(tup1,tup2):
    eu_dist = np.linalg.norm(np.array(tup1) - np.array(tup2))
    return eu_dist

def c_k(d,dist,k,r):
    c = main(d,dist,k,r)
    temp_v = 'v'
    temp_b = 'b'
    centroids = [a[temp_v] for key, a in c.items() if temp_v in a]
    clusters = [a[temp_b] for key, a in c.items() if temp_b in a]
    return centroids,clusters

def main(d,dist,k,thresh): 
    i = 0
    j = 1
    idx = 0
    prev_cen = list()
    tow = list()
    compare = float('inf')
    dist =list()
    cen_values= defaultdict(dict)
    cen_values['j'] = defaultdict(dict)
    c = cen_values['j']
    rand_centroids = random.choices(d,k=k)
    for j in range(1,k+1):
        c[j]= {'v': rand_centroids[j-1], 'b':list()}
    while(True):
        tow = list()
        for data_point in d:
            j=1
            while(j<=k):
                dist.append(eud(c[j]['v'],data_point))
                j+=1
            idx = dist.index(np.min(dist)) 
            c[idx+1]['b'].append(data_point)
            dist = list()
        if compare < thresh:
            break
        for j in range(1,k+1):
            n = len(c[j]['b'])
            prev_cen = c[j]['v']
            if n!=0:
                c[j]['v'] = (*map(lambda x: x/n, (*map(sum, zip(*c[j]['b'])),)),)
            c[j]['b']=list()
            diff_c = np.subtract(prev_cen,c[j]['v'])
            tow.append(math.sqrt(abs((np.dot(tuple(reversed(diff_c)),diff_c)))/k))
        compare = sum(tow)  
    return c


# In[27]:


prad_class= rna_merged_df[rna_merged_df.Class=='PRAD']
luad_class=rna_merged_df[rna_merged_df.Class=='LUAD']
brca_class= rna_merged_df[rna_merged_df.Class=='BRCA']
kirc_class=rna_merged_df[rna_merged_df.Class=='KIRC']
coad_class= rna_merged_df[rna_merged_df.Class=='COAD']

prad_class=prad_class.drop('Class', axis=1)
luad_class=luad_class.drop('Class', axis=1)
brca_class=brca_class.drop('Class', axis=1)
kirc_class=kirc_class.drop('Class', axis=1)
coad_class=coad_class.drop('Class', axis=1)


# In[28]:


set_prad= set(map(tuple,np.array(prad_class)))
set_luad= set(map(tuple,np.array(luad_class)))
set_brca= set(map(tuple,np.array(brca_class)))
set_kirc= set(map(tuple,np.array(kirc_class)))
set_coad= set(map(tuple,np.array(coad_class)))

for k in range(5,6):
    total_error = list()
    dist=list()
    for _ in range(0,20):
        error = []
        sum_error =[]
        centroids,clusters = c_k(np.array(rna_merged_df)[:,:-1],dist,k,10) 
        prad = np.zeros(shape = (1,k))
        luad = np.zeros(shape = (1,k))
        brca = np.zeros(shape = (1,k))
        kirc = np.zeros(shape = (1,k))
        coad = np.zeros(shape = (1,k))
        for s in range(0,k):
            for p in range(0,len(clusters[s])):
                if tuple(clusters[s][p]) in set_prad:
                    prad[0][s]+=1
                if tuple(clusters[s][p]) in set_luad:
                    luad[0][s]+=1
                if tuple(clusters[s][p]) in set_brca:
                    brca[0][s]+=1
                if tuple(clusters[s][p]) in set_kirc:
                    kirc[0][s]+=1
                if tuple(clusters[s][p]) in set_coad:
                    coad[0][s]+=1  
            if (prad[0][s]+luad[0][s]+brca[0][s]+kirc[0][s]+coad[0][s])!=0:
                temp = np.max([prad[0][s],luad[0][s],brca[0][s],kirc[0][s],coad[0][s]])/(prad[0][s]+luad[0][s]+brca[0][s]+kirc[0][s]+coad[0][s])
            else:
                temp = 0
            error.append(temp)
        sum_error  = sum(error)/k
        total_error.append(sum_error)
    plt.boxplot(total_error)
    plt.xlabel(f'cluster K= {k}')
    plt.ylabel('Error rate')
    plt.title('K means Error Boxplot of RNA gene data')
    plt.savefig(f'Boxplot c_k {k} of RNA gene data')
    plt.clf()


# In[29]:


scaled_pca_deltar = pd.DataFrame(pca_deltar)
scaled_pca_deltar['Class']= rna_merged_df.Class


# In[30]:


prad_class= scaled_pca_deltar[scaled_pca_deltar.Class=='PRAD']
luad_class=scaled_pca_deltar[scaled_pca_deltar.Class=='LUAD']
brca_class= scaled_pca_deltar[scaled_pca_deltar.Class=='BRCA']
kirc_class=scaled_pca_deltar[scaled_pca_deltar.Class=='KIRC']
coad_class= scaled_pca_deltar[scaled_pca_deltar.Class=='COAD']

prad_class=prad_class.drop('Class', axis=1)
luad_class=luad_class.drop('Class', axis=1)
brca_class=brca_class.drop('Class', axis=1)
kirc_class=kirc_class.drop('Class', axis=1)
coad_class=coad_class.drop('Class', axis=1)


# In[31]:


set_prad= set(map(tuple,np.array(prad_class)))
set_luad= set(map(tuple,np.array(luad_class)))
set_brca= set(map(tuple,np.array(brca_class)))
set_kirc= set(map(tuple,np.array(kirc_class)))
set_coad= set(map(tuple,np.array(coad_class)))

for k in range(5,6):
    total_error = list()
    dist=list()
    for _ in range(0,20):
        error = []
        sum_error =[]
        centroids,clusters = c_k(np.array(scaled_pca_deltar)[:,:-1],dist,k,10) 
        prad = np.zeros(shape = (1,k))
        luad = np.zeros(shape = (1,k))
        brca = np.zeros(shape = (1,k))
        kirc = np.zeros(shape = (1,k))
        coad = np.zeros(shape = (1,k))
        for s in range(0,k):
            for p in range(0,len(clusters[s])):
                if tuple(clusters[s][p]) in set_prad:
                    prad[0][s]+=1
                if tuple(clusters[s][p]) in set_luad:
                    luad[0][s]+=1
                if tuple(clusters[s][p]) in set_brca:
                    brca[0][s]+=1
                if tuple(clusters[s][p]) in set_kirc:
                    kirc[0][s]+=1
                if tuple(clusters[s][p]) in set_coad:
                    coad[0][s]+=1  
            if (prad[0][s]+luad[0][s]+brca[0][s]+kirc[0][s]+coad[0][s])!=0:
                temp = np.max([prad[0][s],luad[0][s],brca[0][s],kirc[0][s],coad[0][s]])/(prad[0][s]+luad[0][s]+brca[0][s]+kirc[0][s]+coad[0][s])
            else:
                temp = 0
            error.append(temp)
        sum_error  = sum(error)/k
        total_error.append(sum_error)
    plt.boxplot(total_error)
    plt.xlabel(f'cluster K= {k}')
    plt.ylabel('Error rate')
    plt.title('K means Error Boxplot of RNA gene data for 90 percent variance data')
    plt.savefig(f'Boxplot c_k {k} of RNA gene data for 90 percent variance data')
    plt.clf()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# In[2]:


# PCA algorithm 
import numpy as np
from scipy.linalg import svd 

def pca(dim,inp_matrix):
    length = len(inp_matrix)
    delta_hat= np.empty((length,0))
    identity_matrix = np.identity(length)
    e_vector = np.ones(length)
    delta_tilde= np.dot((identity_matrix-((np.multiply(e_vector,e_vector.reshape(length,1)))/length)),inp_matrix)
    u_leftVector,singular_val,v_rightVectorTranspose = svd(delta_tilde)
    for i in range(0,dim):
        arr= np.dot(singular_val[i],u_leftVector[:,i]).reshape(len(u_leftVector),1)
        delta_hat = np.append(delta_hat,arr,axis = 1)

    return delta_hat,u_leftVector,singular_val,v_rightVectorTranspose,delta_tilde


# In[3]:


import pandas as pd
import urllib.request
import tarfile
import io

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz"


response = urllib.request.urlopen(url)
tar_file = io.BytesIO(response.read())
tar = tarfile.open(fileobj=tar_file)
rna_seq_df= pd.read_csv(tar.extractfile("TCGA-PANCAN-HiSeq-801x20531/data.csv"))
label_df = pd.read_csv(tar.extractfile("TCGA-PANCAN-HiSeq-801x20531/labels.csv"))
rna_merged_df= pd.merge(rna_seq_df, label_df, on='Unnamed: 0')
rna_merged_df = rna_merged_df.drop(columns=['Unnamed: 0'])


# In[4]:


prad_class= rna_merged_df[rna_merged_df.Class=='PRAD']
luad_class=rna_merged_df[rna_merged_df.Class=='LUAD']
brca_class= rna_merged_df[rna_merged_df.Class=='BRCA']
kirc_class=rna_merged_df[rna_merged_df.Class=='KIRC']
coad_class= rna_merged_df[rna_merged_df.Class=='COAD']


# In[5]:


# creating 10 samples of each class
prad_sample = prad_class.sample(n=10, random_state= 2500)
luad_sample = luad_class.sample(n=10,random_state= 2500)
brca_sample = brca_class.sample(n=10,random_state= 2500)
kirc_sample = kirc_class.sample(n=10,random_state= 2500)
coad_sample = coad_class.sample(n=10,random_state= 2500)

sample_50_df= pd.concat([prad_sample,luad_sample,brca_sample,kirc_sample,coad_sample])


# In[6]:


# 50 samples from data
sample_50_df.shape


# In[7]:


sample_50_df


# ## 1

# In[31]:


# hierarchical clustering with complete linkage

complete_linkage= linkage(np.array(sample_50_df)[:,:-1], method="complete", metric="euclidean")
plt.figure(figsize=(10, 5))
plt.xlabel("Data index")
plt.ylabel("Dendogram height")
plt.title("Dendogram")
dend=dendrogram(complete_linkage, labels=sample_50_df.index, orientation="top")
plt.savefig('dendogram 1',bbox_inches='tight')
plt.show()


# ## 2

# In[11]:


height_cut_val = 300
clusters = fcluster(complete_linkage, height_cut_val, criterion='distance')


# In[12]:


clusters


# In[13]:


dp_index = np.array(dend['ivl'])
lbls_dp = np.append(clusters.reshape(50,1),dp_index.reshape(50,1),axis = 1)


# In[14]:


lbls_dp


# In[15]:


df_changed = pd.DataFrame(lbls_dp)
df_changed = df_changed.replace({0: {1: 'PRAD', 5: 'BRCA',2:'KIRC',3:'COAD',4:'LUAD'}})
lbls_dp = df_changed.to_numpy()


# In[16]:


# error rate for 5 clusters without pca
true = 0
false = 0
for f in range(0,len(sample_50_df)):
    if (lbls_dp[f][0]) == (sample_50_df.loc[lbls_dp[f][1]].Class):
        true+=1
    else:
        false+=1

tot=false+true
error = false/tot
error


# In[17]:


silhouette_avg = silhouette_score(np.array(sample_50_df)[:,:-1], clusters)

# print silhouette score
print("Silhouette score:", silhouette_avg)


# ## 3
# 

# In[18]:


pca_samp50_data, u_leftVector, singular_val, v_rightVectorTranspose, delta_tilde = pca(len(sample_50_df),np.array(sample_50_df.iloc[:,:-1]))


# In[19]:


perct_variance = (singular_val ** 2) / np.sum(singular_val ** 2)


# In[20]:


for i in range(0,len(perct_variance)):
    total_var1 = sum(perct_variance[:i])*100
    if (total_var1 >= 90):
        print(f"The number of components where percentage variance of total 90 percent is reached is {i}")
        break


# In[21]:


pca_samp50_30= pca_samp50_data[:,:30]


# In[22]:


pca_samp50_30.shape


# In[33]:


# heirarchical clustering with complete linkage with pca

complete_linkage_pcareduced= linkage(pca_samp50_30, method="complete", metric="euclidean")
plt.figure(figsize=(10, 5))
plt.xlabel("Data index")
plt.ylabel("Dendogram height")
plt.title("Dendogram")
dend2=dendrogram(complete_linkage_pcareduced,labels=sample_50_df.index, orientation="top")
plt.savefig('dendogram 2',bbox_inches='tight')
plt.show()


# ### 4

# In[24]:


height_cut_value = 300
clusters = fcluster(complete_linkage_pcareduced, height_cut_value, criterion='distance')


# In[25]:


clusters


# In[26]:


dp_index = np.array(dend2['ivl'])
lbls_dp = np.append(clusters.reshape(50,1),dp_index.reshape(50,1),axis = 1)


# In[27]:


lbls_dp


# In[28]:


df_changed = pd.DataFrame(lbls_dp)
df_changed = df_changed.replace({0: {1: 'PRAD', 5: 'BRCA',2:'KIRC',3:'COAD',4:'LUAD'}})
lbls_dp = df_changed.to_numpy()


# In[29]:


# error rate for 50 sampled data with pca
true = 0
false = 0
for f in range(0,len(sample_50_df)):
    if (lbls_dp[f][0]) == (sample_50_df.loc[lbls_dp[f][1]].Class):
        true+=1
    else:
        false+=1

tot=false+true
error = false/tot
error


# In[30]:


from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(pca_samp50_30, clusters)

# print silhouette score
print("Silhouette score:", silhouette_avg)


#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder


# In[52]:


# PCA algorithm 
import numpy as np
from scipy.linalg import svd 

def pca(dim,inp_matrix):
    length = len(inp_matrix)
    delta_hat= np.empty((length,0))
    identity_matrix = np.identity(length)
    e_vector = np.ones(length)
    delta_tilde= np.dot((identity_matrix-((np.multiply(e_vector,e_vector.reshape(length,1)))/length)),inp_matrix)
    u_leftVector,singular_val,v_rightVectorTranspose = svd(delta_tilde)
    for i in range(0,dim):
        arr= np.dot(singular_val[i],u_leftVector[:,i]).reshape(len(u_leftVector),1)
        delta_hat = np.append(delta_hat,arr,axis = 1)

    return delta_hat,u_leftVector,singular_val,v_rightVectorTranspose,delta_tilde


# In[53]:


import pandas as pd
import urllib.request
import tarfile
import io

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz"


response = urllib.request.urlopen(url)
tar_file = io.BytesIO(response.read())
tar = tarfile.open(fileobj=tar_file)
rna_seq_df= pd.read_csv(tar.extractfile("TCGA-PANCAN-HiSeq-801x20531/data.csv"))
label_df = pd.read_csv(tar.extractfile("TCGA-PANCAN-HiSeq-801x20531/labels.csv"))


# In[54]:


rna_merged_df= pd.merge(rna_seq_df, label_df, on='Unnamed: 0')
rna_merged_df = rna_merged_df.drop(columns=['Unnamed: 0'])


# ### kmeans without pca

# In[55]:


scaled_rna_data = StandardScaler().fit_transform(rna_merged_df.iloc[:,:-1].values)
rna_merged_df.iloc[:,:-1] = pd.DataFrame(scaled_rna_data, index=rna_merged_df.iloc[:,:-1].index, columns=rna_merged_df.iloc[:,:-1].columns)


# In[6]:


#kmeans algorithm ck
import numpy as np
import random
import math

def eud(tup1,tup2):
    eu_dist = np.linalg.norm(np.array(tup1) - np.array(tup2))
    return eu_dist

def c_k(d,dist,k,r):
    c = main(d,dist,k,r)
    temp_v = 'v'
    temp_b = 'b'
    centroids = [a[temp_v] for key, a in c.items() if temp_v in a]
    clusters = [a[temp_b] for key, a in c.items() if temp_b in a]
    return centroids,clusters

def main(d,dist,k,thresh): 
    i = 0
    j = 1
    idx = 0
    prev_cen = list()
    tow = list()
    compare = float('inf')
    dist =list()
    cen_values= defaultdict(dict)
    cen_values['j'] = defaultdict(dict)
    c = cen_values['j']
    rand_centroids = random.choices(d,k=k)
    for j in range(1,k+1):
        c[j]= {'v': rand_centroids[j-1], 'b':list()}
    while(True):
        tow = list()
        for data_point in d:
            j=1
            while(j<=k):
                dist.append(eud(c[j]['v'],data_point))
                j+=1
            idx = dist.index(np.min(dist)) 
            c[idx+1]['b'].append(data_point)
            dist = list()
        if compare < thresh:
            break
        for j in range(1,k+1):
            n = len(c[j]['b'])
            prev_cen = c[j]['v']
            if n!=0:
                c[j]['v'] = (*map(lambda x: x/n, (*map(sum, zip(*c[j]['b'])),)),)
            c[j]['b']=list()
            diff_c = np.subtract(prev_cen,c[j]['v'])
            tow.append(math.sqrt(abs((np.dot(tuple(reversed(diff_c)),diff_c)))/k))
        compare = sum(tow)  
    return c


# In[56]:


prad_class= rna_merged_df[rna_merged_df.Class=='PRAD']
luad_class=rna_merged_df[rna_merged_df.Class=='LUAD']
brca_class= rna_merged_df[rna_merged_df.Class=='BRCA']
kirc_class=rna_merged_df[rna_merged_df.Class=='KIRC']
coad_class= rna_merged_df[rna_merged_df.Class=='COAD']

prad_class=prad_class.drop('Class', axis=1)
luad_class=luad_class.drop('Class', axis=1)
brca_class=brca_class.drop('Class', axis=1)
kirc_class=kirc_class.drop('Class', axis=1)
coad_class=coad_class.drop('Class', axis=1)


# In[8]:


set_prad= set(map(tuple,np.array(prad_class)))
set_luad= set(map(tuple,np.array(luad_class)))
set_brca= set(map(tuple,np.array(brca_class)))
set_kirc= set(map(tuple,np.array(kirc_class)))
set_coad= set(map(tuple,np.array(coad_class)))

for k in range(5,6):
    total_error = list()
    dist=list()
    all_labels_list=list()
    for _ in range(0,20):
        error = []
        sum_error =[]
        centroids,clusters = c_k(np.array(rna_merged_df)[:,:-1],dist,k,10) 
        prad = np.zeros(shape = (1,k))
        luad = np.zeros(shape = (1,k))
        brca = np.zeros(shape = (1,k))
        kirc = np.zeros(shape = (1,k))
        coad = np.zeros(shape = (1,k))
        cluster_labels=list()
        for s in range(0,k):
            for p in range(0,len(clusters[s])):
                if tuple(clusters[s][p]) in set_prad:
                    cluster_labels.append(1)
                    prad[0][s]+=1
                if tuple(clusters[s][p]) in set_luad:
                    cluster_labels.append(2)
                    luad[0][s]+=1
                if tuple(clusters[s][p]) in set_brca:
                    cluster_labels.append(3)
                    brca[0][s]+=1
                if tuple(clusters[s][p]) in set_kirc:
                    cluster_labels.append(4)
                    kirc[0][s]+=1
                if tuple(clusters[s][p]) in set_coad:
                    cluster_labels.append(5)
                    coad[0][s]+=1  
            if (prad[0][s]+luad[0][s]+brca[0][s]+kirc[0][s]+coad[0][s])!=0:
                temp = np.max([prad[0][s],luad[0][s],brca[0][s],kirc[0][s],coad[0][s]])/(prad[0][s]+luad[0][s]+brca[0][s]+kirc[0][s]+coad[0][s])
            else:
                temp = 0
            error.append(temp)
        sum_error  = sum(error)/k
        total_error.append(sum_error)
        all_labels_list.append(cluster_labels)
all_labels_list = np.array(all_labels_list)
#     plt.boxplot(total_error)
#     plt.xlabel(f'cluster K= {k}')
#     plt.ylabel('Error rate')
#     plt.title('K means Error Boxplot of RNA gene data without pca')
#     plt.savefig(f'Boxplot c_k {k} of RNA gene data without pca')
#     plt.clf()


# In[57]:


silhouette_coef = list()
for d in range(0,len(all_labels_list)):  
    ss = silhouette_score(rna_merged_df.iloc[:,:-1], all_labels_list[d])
    silhouette_coef.append(ss)

silhouette_coef


# In[59]:


plt.boxplot(silhouette_coef)
plt.title('Boxplot for Kmeans clustering')
plt.ylabel('Silhouette score')
plt.xlabel('Kmeans cluster k=5 based on silhouette score without pca')
plt.savefig('Boxplot for Kmeans clustering based on silhouette score without pca')
plt.show()


# In[60]:


true_labels = rna_merged_df.iloc[:,-1]
label_encoder = LabelEncoder()
true_labels_num = label_encoder.fit_transform(true_labels)

adjusted_randscore_kmeans_withoutpca = []
for d in range(0,len(all_labels_list)):  
    adss = adjusted_rand_score( all_labels_list[d],true_labels_num)
    adjusted_randscore_kmeans_withoutpca.append(adss)

adjusted_randscore_kmeans_withoutpca


# In[61]:


plt.boxplot(adjusted_randscore_kmeans_withoutpca)
plt.title('Boxplot for Kmeans clustering')
plt.ylabel('Adjusted Rand score')
plt.xlabel('Kmeans cluster k=5 based on adjusted rand score without pca')
plt.savefig('Boxplot for Kmeans clustering based on adjusted rand score without pca')
plt.show()


# ### Single linkage without pca

# In[62]:


# silhouette score
single_linkage_mode_without_pca = AgglomerativeClustering(n_clusters=5, metric="euclidean",linkage='single')
single_linkage_mode_without_pca=single_linkage_mode_without_pca.fit(rna_merged_df.iloc[:,:-1]).labels_
silhouette_score_single=silhouette_score(rna_merged_df.iloc[:,:-1],single_linkage_mode_without_pca)
silhouette_score_single


# In[63]:


# Adjusted rand score
true_labels = rna_merged_df.iloc[:,-1]
label_encoder = LabelEncoder()
true_labels_num = label_encoder.fit_transform(true_labels)

adjusted_randscore_single_withoutpca = adjusted_rand_score(true_labels_num, single_linkage_mode_without_pca)
adjusted_randscore_single_withoutpca


# ### Complete linkage without pca

# In[64]:


# silhouette score
complete_linkage_mode_without_pca = AgglomerativeClustering(n_clusters=5, metric="euclidean",linkage='complete')
complete_linkage_mode_without_pca=complete_linkage_mode_without_pca.fit(rna_merged_df.iloc[:,:-1]).labels_
silhouette_score_complete=silhouette_score(rna_merged_df.iloc[:,:-1],complete_linkage_mode_without_pca)


# In[65]:


# Adjusted rand score
true_labels = rna_merged_df.iloc[:,-1]
label_encoder = LabelEncoder()
true_labels_num = label_encoder.fit_transform(true_labels)

adjusted_randscore_complete_withoutpca = adjusted_rand_score(true_labels_num,complete_linkage_mode_without_pca)
adjusted_randscore_complete_withoutpca


# ### Ward linkage without pca
# 

# In[66]:


# silhouette score
ward_linkage_mode_without_pca = AgglomerativeClustering(n_clusters=5, metric="euclidean",linkage='ward')
ward_linkage_mode_without_pca=ward_linkage_mode_without_pca.fit(rna_merged_df.iloc[:,:-1]).labels_
silhouette_score_ward=silhouette_score(rna_merged_df.iloc[:,:-1],ward_linkage_mode_without_pca)


# In[67]:


# Adjusted rand score
true_labels = rna_merged_df.iloc[:,-1]
label_encoder = LabelEncoder()
true_labels_num = label_encoder.fit_transform(true_labels)

adjusted_randscore_ward_withoutpca = adjusted_rand_score(true_labels_num,ward_linkage_mode_without_pca)
adjusted_randscore_ward_withoutpca


# In[87]:


plt.boxplot([silhouette_score_single,silhouette_score_complete,silhouette_score_ward])
plt.title('Boxplot for Hierarchical clustering')
plt.ylabel('Silhouette score')
plt.xlabel('Linkage methods-single,complete,ward')
plt.savefig('Boxplot for Hierarchical clustering')
plt.show()


# In[85]:


plt.boxplot([adjusted_randscore_single_withoutpca,adjusted_randscore_complete_withoutpca,adjusted_randscore_ward_withoutpca])
plt.title('Boxplot for Hierarchical clustering')
plt.ylabel('adjusted rand')
plt.xlabel('Linkage methods-single,complete,ward')
plt.savefig('Boxplot for Hierarchical clustering adjusted rand int without pca')
plt.show()


# ###  kmeans with pca

# In[69]:


pca_all_data, u_leftVector, singular_val, v_rightVectorTranspose, delta_tilde = pca(len(rna_merged_df),np.array(rna_merged_df.iloc[:,:-1]))


# In[70]:


pca_all_data=pd.DataFrame(pca_all_data)
pca_all_data['Class'] = rna_merged_df.Class
pca_all_data


# In[71]:


prad_class= pca_all_data[pca_all_data.Class=='PRAD']
luad_class=pca_all_data[pca_all_data.Class=='LUAD']
brca_class= pca_all_data[pca_all_data.Class=='BRCA']
kirc_class=pca_all_data[pca_all_data.Class=='KIRC']
coad_class= pca_all_data[pca_all_data.Class=='COAD']

prad_class=prad_class.drop('Class', axis=1)
luad_class=luad_class.drop('Class', axis=1)
brca_class=brca_class.drop('Class', axis=1)
kirc_class=kirc_class.drop('Class', axis=1)
coad_class=coad_class.drop('Class', axis=1)


# In[72]:


set_prad= set(map(tuple,np.array(prad_class)))
set_luad= set(map(tuple,np.array(luad_class)))
set_brca= set(map(tuple,np.array(brca_class)))
set_kirc= set(map(tuple,np.array(kirc_class)))
set_coad= set(map(tuple,np.array(coad_class)))

for k in range(5,6):
    total_error = list()
    dist=list()
    all_labels_list=list()
    for _ in range(0,20):
        error = []
        sum_error =[]
        centroids,clusters = c_k(np.array(pca_all_data)[:,:-1],dist,k,10) 
        prad = np.zeros(shape = (1,k))
        luad = np.zeros(shape = (1,k))
        brca = np.zeros(shape = (1,k))
        kirc = np.zeros(shape = (1,k))
        coad = np.zeros(shape = (1,k))
        cluster_labels=list()
        for s in range(0,k):
            for p in range(0,len(clusters[s])):
                if tuple(clusters[s][p]) in set_prad:
                    cluster_labels.append(1)
                    prad[0][s]+=1
                if tuple(clusters[s][p]) in set_luad:
                    cluster_labels.append(2)
                    luad[0][s]+=1
                if tuple(clusters[s][p]) in set_brca:
                    cluster_labels.append(3)
                    brca[0][s]+=1
                if tuple(clusters[s][p]) in set_kirc:
                    cluster_labels.append(4)
                    kirc[0][s]+=1
                if tuple(clusters[s][p]) in set_coad:
                    cluster_labels.append(5)
                    coad[0][s]+=1  
            if (prad[0][s]+luad[0][s]+brca[0][s]+kirc[0][s]+coad[0][s])!=0:
                temp = np.max([prad[0][s],luad[0][s],brca[0][s],kirc[0][s],coad[0][s]])/(prad[0][s]+luad[0][s]+brca[0][s]+kirc[0][s]+coad[0][s])
            else:
                temp = 0
            error.append(temp)
        sum_error  = sum(error)/k
        total_error.append(sum_error)
        all_labels_list.append(cluster_labels)
all_labels_list = np.array(all_labels_list)


# In[73]:


silhouette_coef = list()
for d in range(0,len(all_labels_list)):  
    ss = silhouette_score(pca_all_data.iloc[:,:-1], all_labels_list[d])
    silhouette_coef.append(ss)

silhouette_coef


# In[74]:


plt.boxplot(silhouette_coef)
plt.title('Boxplot for Kmeans clustering with pca')
plt.ylabel('Silhouette score')
plt.xlabel('Kmeans cluster k=5')
plt.savefig('Boxplot for Kmeans clustering with pca')
plt.show()


# In[75]:


true_labels = pca_all_data.iloc[:,-1]
label_encoder = LabelEncoder()
true_labels_num = label_encoder.fit_transform(true_labels)

adjusted_randscore_kmeans_withpca = []
for d in range(0,len(all_labels_list)):  
    adss = adjusted_rand_score( all_labels_list[d],true_labels_num)
    adjusted_randscore_kmeans_withpca.append(adss)

adjusted_randscore_kmeans_withpca


# In[76]:


plt.boxplot(adjusted_randscore_kmeans_withoutpca)
plt.title('Boxplot for Kmeans clustering with pca')
plt.ylabel('Adjusted Rand score')
plt.xlabel('Kmeans cluster k=5 based on adjusted rand score with pca')
plt.savefig('Boxplot for Kmeans clustering based on adjusted rand score with pca')
plt.show()


# ### Single linkage with pca

# In[77]:


# silhouette score
single_linkage_mode_with_pca = AgglomerativeClustering(n_clusters=5, metric="euclidean",linkage='single')
single_linkage_mode_with_pca=single_linkage_mode_with_pca.fit(pca_all_data.iloc[:,:-1]).labels_
silhouette_score_single_withpca=silhouette_score(rna_merged_df.iloc[:,:-1],single_linkage_mode_with_pca)
silhouette_score_single_withpca


# In[78]:


# Adjusted rand score
true_labels = rna_merged_df.iloc[:,-1]
label_encoder = LabelEncoder()
true_labels_num = label_encoder.fit_transform(true_labels)

adjusted_randscore_single_withpca = adjusted_rand_score(true_labels_num, single_linkage_mode_with_pca)
adjusted_randscore_single_withpca


# ### Complete linkage with pca

# In[79]:


# silhouette score
complete_linkage_mode_with_pca = AgglomerativeClustering(n_clusters=5, metric="euclidean",linkage='complete')
complete_linkage_mode_with_pca=complete_linkage_mode_with_pca.fit(pca_all_data.iloc[:,:-1]).labels_
silhouette_score_complete_withpca=silhouette_score(pca_all_data.iloc[:,:-1],complete_linkage_mode_with_pca)
silhouette_score_complete_withpca


# In[80]:


# Adjusted rand score
true_labels = rna_merged_df.iloc[:,-1]
label_encoder = LabelEncoder()
true_labels_num = label_encoder.fit_transform(true_labels)

adjusted_randscore_complete_withpca = adjusted_rand_score(true_labels_num,complete_linkage_mode_with_pca)
adjusted_randscore_complete_withpca


# ### Ward linkage with pca

# In[81]:


# silhouette score
ward_linkage_mode_with_pca = AgglomerativeClustering(n_clusters=5, metric="euclidean",linkage='ward')
ward_linkage_mode_with_pca=ward_linkage_mode_with_pca.fit(pca_all_data.iloc[:,:-1]).labels_
silhouette_score_ward_withpca=silhouette_score(pca_all_data.iloc[:,:-1],ward_linkage_mode_with_pca)
silhouette_score_ward_withpca


# In[82]:


# Adjusted rand score
true_labels = rna_merged_df.iloc[:,-1]
label_encoder = LabelEncoder()
true_labels_num = label_encoder.fit_transform(true_labels)

adjusted_randscore_ward_withpca = adjusted_rand_score(true_labels_num,ward_linkage_mode_with_pca)
adjusted_randscore_ward_withpca


# In[83]:


plt.boxplot([silhouette_score_single_withpca,silhouette_score_complete_withpca,silhouette_score_ward_withpca])
plt.title('Boxplot for Hierarchical clustering with PCA')
plt.ylabel('Silhouette score')
plt.xlabel('Linkage methods-single,complete,ward')
plt.savefig('Boxplot for Hierarchical clustering with PCA')
plt.show()


# In[88]:


plt.boxplot([adjusted_randscore_single_withpca,adjusted_randscore_complete_withpca,adjusted_randscore_ward_withpca])
plt.title('Boxplot for Hierarchical clustering')
plt.ylabel('adjusted rand')
plt.xlabel('Linkage methods-single,complete,ward')
plt.savefig('Boxplot for Hierarchical clustering adjusted rand int with pca')
plt.show()


# In[ ]:





# In[ ]:














