#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np

dataset = tf.keras.datasets.mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = dataset
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


# In[3]:


encoder = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(100, activation='selu'),
    tf.keras.layers.Dense(30, activation='selu')
])
decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation='selu', input_shape=[30]),
    tf.keras.layers.Dense(28*28, activation='sigmoid'),
    tf.keras.layers.Reshape([28, 28])
])
ae = tf.keras.models.Sequential([encoder, decoder])
ae.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
ae.fit(X_train, X_train, epochs=20)
ae.evaluate(X_test, X_test, return_dict=True)


# In[4]:


import matplotlib.pyplot as plt
def plot_reconstructions(model, images=X_test, n_images=5):
    reconstructions = np.clip(model.predict(images[:n_images]), 0, 1)
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plt.imshow(images[image_index], cmap="binary")
        plt.axis("off")
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plt.imshow(reconstructions[image_index], cmap="binary")
        plt.axis("off")
plot_reconstructions(ae)


# In[5]:


ae.save('ae_stacked.keras')


# In[6]:


from tensorflow import keras
conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    keras.layers.Conv2D(16, kernel_size=3, padding="SAME",
    activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=3, padding="SAME",
    activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(64, kernel_size=3, padding="SAME",
    activation="selu"),
    keras.layers.MaxPool2D(pool_size=2)
])
conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2,
    padding="VALID", activation="selu",
    input_shape=[3, 3, 64]),
    keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2,
    padding="SAME", activation="selu"),
    keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2,
    padding="SAME",
    activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
conv_ae = tf.keras.models.Sequential([conv_encoder, conv_decoder])
conv_ae.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
conv_ae.fit(X_train, X_train, epochs=5)
conv_ae.evaluate(X_test, X_test, return_dict=True)


# In[9]:


plot_reconstructions(conv_ae)


# In[8]:


conv_ae.save('ae_conv.keras')


# In[12]:


from sklearn.manifold import TSNE
X_valid_compressed = conv_encoder.predict(X_valid)
X_valid_flat = X_valid_compressed.reshape((X_valid_compressed.shape[0], -1))
tsne = TSNE(init="pca", learning_rate="auto", random_state=42)
X_valid_2D = tsne.fit_transform(X_valid_flat)
plt.figure(figsize=(10, 5))
plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap="tab10")
plt.show()


# In[13]:


import matplotlib as mpl
plt.figure(figsize=(10, 5))
cmap = plt.cm.tab10
Z = X_valid_2D
Z = (Z - Z.min()) / (Z.max() - Z.min()) # normalize to the 0-1 range
plt.scatter(Z[:, 0], Z[:, 1], c=y_valid, s=10, cmap=cmap)
image_positions = np.array([[1., 1.]])
for index, position in enumerate(Z):
    dist = ((position - image_positions) ** 2).sum(axis=1)
    if dist.min() > 0.02: # if far enough from other images
        image_positions = np.r_[image_positions, [position]]
        imagebox = mpl.offsetbox.AnnotationBbox(
            mpl.offsetbox.OffsetImage(X_valid[index], cmap="binary"),
            position, bboxprops={"edgecolor": cmap(y_valid[index]), "lw": 2})
        plt.gca().add_artist(imagebox)
plt.axis("off")
plt.show()


# In[14]:


dropout_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu")
])
dropout_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
dropout_ae = tf.keras.models.Sequential([dropout_encoder, dropout_decoder])
dropout_ae.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
dropout_ae.fit(X_train, X_train, epochs=5)
dropout_ae.evaluate(X_test, X_test, return_dict=True)


# In[15]:


dropout = keras.layers.Dropout(0.3)
plot_reconstructions(dropout_ae, dropout(X_valid, training=True))


# In[16]:


dropout_ae.save('ae_denoise.keras')

