#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request
import tarfile
import gzip
import shutil
import os

os.makedirs("data", exist_ok=True)
 

url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
urllib.request.urlretrieve(url, "data/housing.tgz")

with tarfile.open("data/housing.tgz", "r:gz") as tar:
    tar.extractall("data")

with open("data/housing.csv", "rb") as f_in:
    with gzip.open("data/housing.csv.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

os.remove("data/housing.tgz")


# In[2]:


import pandas as pd
df = pd.read_csv('data/housing.csv.gz')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df["ocean_proximity"].value_counts()
df["ocean_proximity"].describe()


# In[7]:


import matplotlib.pyplot as plt

df.hist(bins=50, figsize=(20,15))
plt.show()
plt.savefig("obraz1.png")


# In[8]:


df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(7,4))
plt.savefig("obraz2.png")


# In[9]:


df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(7,3), 
        s=df["population"]/100, label="population", 
        c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.savefig("obraz3.png")


# In[10]:


df_numeric = df.select_dtypes(include=[float, int])
df_numeric.corr()["median_house_value"].sort_values(ascending=False)


# In[11]:


df_numeric = df.select_dtypes(include=[float, int])
df_numeric.corr()["median_house_value"].reset_index().rename(columns={"index": "atrybut", "median_house_value": "wspolczynnik_korelacji"}).to_csv("korelacja.csv", index=False)


# In[12]:


import seaborn as sns
sns.pairplot(df)


# In[13]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(train_set), len(test_set)


# In[14]:


train_set.head()


# In[15]:


test_set.head()


# In[16]:


train_corr = train_set.select_dtypes(include=[float, int]).corr()["median_house_value"].sort_values(ascending=False)
test_corr = test_set.select_dtypes(include=[float, int]).corr()["median_house_value"].sort_values(ascending=False)

print("Macierz korelacji - zbiór treningowy:")
print(train_corr)

print("\nMacierz korelacji - zbiór testowy:")
print(test_corr)


# In[17]:


import pickle
train_set.to_pickle("train_set.pkl")
test_set.to_pickle("test_set.pkl")

