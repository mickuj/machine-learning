#!/usr/bin/env python
# coding: utf-8

# In[20]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
print(data_breast_cancer['DESCR'])


# In[21]:


X = data_breast_cancer.data[['mean texture', 'mean symmetry']]
y = data_breast_cancer.target


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

best_f1 = 0
best_depth = None
best_clf = None
results = []
for depth in range(1, 11):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    results.append((depth, f1_train, f1_test, acc_train, acc_test))

    if f1_test > best_f1:
        best_f1 = f1_test
        best_depth = depth
        best_clf = clf

best_depth


# In[24]:


from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(
    best_clf,
    out_file=None,
    feature_names=X.columns,
    class_names=data_breast_cancer.target_names,
    filled=True,
    rounded=True)
graph = graphviz.Source(dot_data)
graph.render('bc', format='png', cleanup=True)


# In[25]:


import pickle

best_result = [best_depth, best_f1, f1_score(y_test, best_clf.predict(X_test)), accuracy_score(y_train, best_clf.predict(X_train)), accuracy_score(y_test, best_clf.predict(X_test))]

with open('f1acc_tree.pkl', 'wb') as f:
    pickle.dump(best_result, f)

best_result


# In[26]:


import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')


# In[27]:


X_ = df[['x']]
y_ = df['y']
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state=42)


# In[29]:


from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

best_mse = float('inf')
best_depth = None
best_reg = None
results = []
for depth in range(1, 16):
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    reg.fit(X_train, y_train)
    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    results.append((depth, mse_train, mse_test))

    if mse_test < best_mse:
        best_mse = mse_test
        best_depth = depth
        best_reg = reg

best_depth


# In[31]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X_, y_, label='dane z df', color='green', alpha=0.5)
X_range = np.linspace(X_.min(), X_.max(), 500).reshape(-1, 1)
y_pred_range = best_reg.predict(X_range)

plt.plot(X_range, y_pred_range, color='blue', label='drzewo decyzyjne')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


# In[32]:


dot_data = export_graphviz(
    best_reg,
    out_file=None,
    feature_names=['x'],
    filled=True,
    rounded=True)
graph = graphviz.Source(dot_data)
graph.render('reg', format='png', cleanup=True)


# In[33]:


best_results = [best_depth, mean_squared_error(y_train, best_reg.predict(X_train)), mean_squared_error(y_test, best_reg.predict(X_test))]

with open('mse_tree.pkl', 'wb') as f:
    pickle.dump(best_results, f)

best_results

