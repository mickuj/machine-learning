#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4

import pandas as pd
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')


# In[4]:


from sklearn.model_selection import train_test_split
X = X.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsRegressor

knn_3_reg = KNeighborsRegressor(n_neighbors=3)
knn_3_reg.fit(X_train, y_train)
knn_5_reg = KNeighborsRegressor(n_neighbors=5)
knn_5_reg.fit(X_train, y_train)

from sklearn.preprocessing import PolynomialFeatures

poly_models = {}
mse_results = {}

for degree in range (2, 6):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)
    poly_models[degree] = (poly_reg, poly_features)


# In[11]:


from sklearn.metrics import mean_squared_error
import pickle
mse_results["lin_reg"] = (
    mean_squared_error(y_train, lin_reg.predict(X_train)),
    mean_squared_error(y_test, lin_reg.predict(X_test))
)
mse_results["knn_3_reg"] = (
    mean_squared_error(y_train, knn_3_reg.predict(X_train)),
    mean_squared_error(y_test, knn_3_reg.predict(X_test))
)
mse_results["knn_5_reg"] = (
    mean_squared_error(y_train, knn_5_reg.predict(X_train)),
    mean_squared_error(y_test, knn_5_reg.predict(X_test))
)
for degree in range(2, 6):
    poly_reg, poly_features = poly_models[degree]
    mse_results[f"poly_{degree}_reg"] = (
        mean_squared_error(y_train, poly_reg.predict(poly_features.transform(X_train))),
        mean_squared_error(y_test, poly_reg.predict(poly_features.transform(X_test)))
    )

for model, (train_mse, test_mse) in mse_results.items():
    print(f"{model}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")

mse_df = pd.DataFrame.from_dict(mse_results, orient="index", columns=["train_mse", "test_mse"])
mse_df.to_pickle("mse.pkl")


# In[14]:


regressor_list = [
    (lin_reg, None),
    (knn_3_reg, None),
    (knn_5_reg, None) ] + [(poly_models[d][0], poly_models[d][1]) for d in range(2, 6)]

with open ("reg.pkl", "wb") as f:
    pickle.dump(regressor_list, f)


# In[ ]:




