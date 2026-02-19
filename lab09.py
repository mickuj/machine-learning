#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# In[2]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[3]:


import matplotlib.pyplot as plt
plt.figure(figsize = (2,2))
plt.imshow(X_train[42], cmap="binary")
plt.axis('off')
plt.show()


# In[4]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
"sandał", "koszula", "półbut", "torba", "but"]
class_names[y_train[42]]


# In[5]:


from tensorflow import keras
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()


# In[6]:


model.compile(loss="sparse_categorical_crossentropy",
optimizer="sgd",
metrics=["accuracy"])


# In[7]:


import os
import time
root_logdir = os.path.join(os.curdir, "image_logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()


# In[8]:


tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)


# In[9]:


history = model.fit(X_train, y_train, epochs=20, validation_split=0.1, callbacks=[tensorboard_cb])


# In[10]:


import numpy as np
image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.figure(figsize = (2,2))
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()


# In[11]:


model.save('fashion_clf.keras')


# In[12]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
housing = fetch_california_housing()


# In[13]:


X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2)


# In[14]:


normalizer = keras.layers.Normalization()
normalizer.adapt(X_train_val)

model1 = keras.models.Sequential([
    normalizer,
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(1)
])


# In[15]:


model1.compile(loss="mse",
optimizer="adam",
metrics=[keras.metrics.RootMeanSquaredError()])


# In[16]:


es = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01, verbose=1)
root_logdir = os.path.join(os.curdir, "housing_logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir1 = get_run_logdir()
tensorboard_cb1 = tf.keras.callbacks.TensorBoard(run_logdir1)


# In[17]:


history1 = model1.fit(
    X_train_val, y_train_val,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[es, tensorboard_cb1]
)


# In[18]:


model1.save('reg_housing_1.keras')


# In[19]:


model2 = keras.models.Sequential([
    normalizer,
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1)
])

model2.compile(loss="mse",
optimizer="adam",
metrics=[keras.metrics.RootMeanSquaredError()])
run_logdir2 = get_run_logdir()
tensorboard_cb2 = tf.keras.callbacks.TensorBoard(run_logdir2)
history2 = model2.fit(
    X_train_val, y_train_val,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[es, tensorboard_cb2]
)


# In[20]:


model2.save('reg_housing_2.keras')


# In[21]:


model3 = keras.models.Sequential([
    normalizer,
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])

model3.compile(loss="mse",
optimizer="adam",
metrics=[keras.metrics.RootMeanSquaredError()])
run_logdir3 = get_run_logdir()
tensorboard_cb3 = tf.keras.callbacks.TensorBoard(run_logdir3)
history3 = model3.fit(
    X_train_val, y_train_val,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[es, tensorboard_cb3]
)


# In[22]:


model3.save('reg_housing_3.keras')

