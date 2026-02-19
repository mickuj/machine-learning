#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow_datasets as tfds
[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
    "tf_flowers",
    split=['train[:10%]', "train[10%:25%]", "train[25%:]"],
    as_supervised=True,
    with_info=True)


# In[2]:


class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples


# In[3]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
index = 0
sample_images = train_set_raw.take(9)
for image, label in sample_images:
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis("off")
plt.show(block=False)


# In[4]:


import tensorflow as tf
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    return resized_image, label

batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[5]:


plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
print(sample_batch)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index]/255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")
plt.show()


# In[6]:


from functools import partial
from tensorflow import keras

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")
model = keras.models.Sequential([
    keras.Input(shape=(224, 224, 3)),
    keras.layers.Rescaling(1./255),
    DefaultConv2D(filters=32),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=64),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=n_classes, activation='softmax')])


# In[7]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_set, validation_data=valid_set, epochs=10)


# In[8]:


import pickle

acc_train = model.evaluate(train_set, verbose=0)[1]
acc_valid = model.evaluate(valid_set, verbose=0)[1]
acc_test = model.evaluate(test_set, verbose=0)[1]

with open('simple_cnn_acc.pkl', 'wb') as f:
    pickle.dump((acc_train, acc_valid, acc_test), f)

acc_train, acc_valid, acc_test 


# In[9]:


model.save('simple_cnn_flowers.keras')


# In[10]:


def preprocessnew(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocessnew_input(resized_image)
    return final_image, label


# In[11]:





# In[12]:


base_model = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[13]:


tf.keras.utils.plot_model(base_model)


# In[14]:


for layer in base_model.layers:
    layer.trainable = False

avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)

optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer, 
    metrics=["accuracy"]
)
model.fit(train_set, validation_data=valid_set, epochs=5)

for layer in base_model.layers:
    layer.trainable = True

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer, 
    metrics=["accuracy"]
)
model.fit(train_set, validation_data=valid_set, epochs=20)


# In[15]:


acc_train = model.evaluate(train_set, verbose=0)[1]
acc_valid = model.evaluate(valid_set, verbose=0)[1]
acc_test = model.evaluate(test_set, verbose=0)[1]

with open('xception_acc.pkl', 'wb') as f:
    pickle.dump((acc_train, acc_valid, acc_test), f)

acc_train, acc_valid, acc_test


# In[16]:


model.save("xception_flowers.keras")

