#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)


# In[2]:


import numpy as np
print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[3]:


from matplotlib import pyplot as plt
pixels = np.array(mnist.data.loc[42]).reshape(28, 28)
plt.imshow(pixels, cmap='gray')
plt.show()


# In[4]:


import pandas as pd

X, y = mnist.data, mnist.target
y = y.astype(int)
data_sorted = pd.DataFrame(X)
data_sorted['label'] = y
data_sorted = data_sorted.sort_values(by='label')


# In[5]:


X_sorted = data_sorted.drop(columns=['label']).values
y_sorted = data_sorted['label'].values


# In[6]:


X_train, X_test = X_sorted[:56000], X_sorted[56000:]
y_train, y_test = y_sorted[:56000], y_sorted[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[7]:


y_train_classes = np.unique(y_train)
y_test_classes = np.unique(y_test)


# In[8]:


print('y_train:', y_train_classes)
print('y_test:', y_test_classes)


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[10]:


y_train_classes = np.unique(y_train)
y_test_classes = np.unique(y_test)
print('y_train:', y_train_classes)
print('y_test:', y_test_classes)


# In[11]:


y_bin_train = (y_train == 0).astype(int)
y_bin_test = (y_test == 0).astype(int)


# In[12]:


from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_bin_train)


# In[13]:


from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_bin_train, sgd.predict(X_train))
test_acc = accuracy_score(y_bin_test, sgd.predict(X_test))


# In[14]:


import pickle
acc_results = [train_acc, test_acc]
with open("sgd_acc.pkl", "wb") as f:
    pickle.dump(acc_results, f)


# In[15]:


print("Dokładność na zbiorze uczącym:", train_acc)
print("Dokładność na zbiorze testowym:", test_acc)


# In[18]:


from sklearn.model_selection import cross_val_score
cross_val = cross_val_score(sgd, X_train, y_bin_train, cv=3, scoring="accuracy")
with open("sgd_cva.pkl", "wb") as f:
    pickle.dump(cross_val, f)


# In[20]:


print("Wyniki walidacji krzyżowej:", cross_val)


# In[21]:


sgd_all = SGDClassifier(random_state=42)
sgd_all.fit(X_train, y_train)


# In[23]:


from sklearn.metrics import confusion_matrix
y_pred = sgd_all.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)


# In[24]:


with open("sgd_cmx.pkl", "wb") as f:
    pickle.dump(matrix, f)


# In[26]:


print("Macierz błędów dla zbioru testowego:")
print(matrix)


# In[ ]:




