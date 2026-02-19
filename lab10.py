#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# In[11]:


import tensorflow as tf

def build_model(n_hidden, n_neurons, optimizer, learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=X_train.shape[1:]))

    for i in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

    if optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "nesterov":
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError("nieznany optymalizator")

    model.compile(loss="mse", optimizer=opt)
    return model


# In[12]:


learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
histories = []

for lr in learning_rates:
    model = build_model(n_hidden=1, n_neurons=30, optimizer="adam", learning_rate=lr)
    history = model.fit(X_train, y_train,
                        epochs=40,
                        validation_data=(X_valid, y_valid),
                        verbose=0)
    histories.append(history)


# In[13]:


import matplotlib.pyplot as plt

for i, history in enumerate(histories):
    plt.plot(history.history["loss"], label=f"Training loss {learning_rates[i]:.0e}")

plt.xlabel("Epoka")
plt.ylabel("Strata (loss)")
plt.legend()
plt.grid(True)
plt.show()


# In[14]:


from scikeras.wrappers import KerasRegressor

es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)
keras_reg = KerasRegressor(build_model, callbacks=[es])


# In[15]:


from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

param_distribs = {
    "model__n_hidden": [0, 1, 2, 3],
    "model__n_neurons": np.arange(1, 101),
    "model__learning_rate": reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
    "model__optimizer": ["sgd", "nesterov", "adam"]
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=5, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), verbose=0)


# In[16]:


rnd_search_cv.best_params_


# In[17]:


import pickle
with open('rnd_search_params.pkl', 'wb') as f:
    pickle.dump(rnd_search_cv.best_params_, f)

with open('rnd_search_scikeras.pkl', 'wb') as f:
    pickle.dump(rnd_search_cv, f)


# In[18]:


import keras_tuner as kt

def build_model_kt(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=3, default=2)
    n_neurons = hp.Int("n_neurons", min_value=1, max_value=100)
    learning_rate = hp.Float("learning_rate", min_value=3e-4, max_value=3e-2, sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "nesterov","adam"])

    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "nesterov":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())

    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))

    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])
    return model


# In[19]:


random_search_tuner = kt.RandomSearch(
build_model_kt, objective="val_mse", max_trials=10, overwrite=True, directory="my_california_housing", project_name="my_rnd_search", seed=42)


# In[20]:


import os

root_logdir = os.path.join(random_search_tuner.project_dir, 'tensorboard')
tb = tf.keras.callbacks.TensorBoard(root_logdir)
es_cb = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0)


# In[21]:


random_search_tuner.search(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),callbacks=[tb, es_cb], verbose=1)


# In[22]:


best_params = random_search_tuner.get_best_hyperparameters(1)[0].values
with open("kt_search_params.pkl", "wb") as f:
    pickle.dump(best_params, f)

best_model = random_search_tuner.get_best_models(1)[0]
best_model.save("kt_best_model.keras")


# In[23]:


best_params

