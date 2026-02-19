#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

tf.keras.utils.get_file(
    "bike_sharing_dataset.zip",
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    cache_dir=".",
    extract=True
)


# In[2]:


import pandas as pd
df = pd.read_csv('datasets/bike_sharing_dataset_extracted/hour.csv')
df['datetime'] = pd.to_datetime(
    df['dteday'] + ' ' + df['hr'].astype(str).str.zfill(2),
    format='%Y-%m-%d %H'
)
df.set_index('datetime', inplace=True)
df


# In[3]:


print((df.index.min(), df.index.max()))


# In[4]:


(365 + 366) * 24 - len(df)


# In[5]:


df_rentals = df[['casual', 'registered', 'cnt']].resample('H').sum().fillna(0)
df_sensors = df[['temp', 'atemp', 'hum', 'windspeed']].resample('H').interpolate()
df_categoricals = df[['holiday', 'weekday', 'workingday', 'weathersit']].resample('H').ffill()
df_new = pd.concat([df_rentals, df_sensors, df_categoricals], axis=1)
df_new


# In[6]:


print(df_new.isna().sum())


# In[7]:


df_new['casual'] /= 1e3
df_new['registered'] /= 1e3
df_new['cnt'] /= 1e3
df_new['weathersit'] /= 4

df_new_2weeks = df_new[:24 * 7 * 2]
df_new_2weeks[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3), title="Dane z 2 tygodni")

df_new_daily = df_new.resample('W').mean()
df_new_daily[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(12, 3), title="Średnie tygodniowe")

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import numpy as np
from sklearn.metrics import mean_absolute_error
import pickle

cnt = df_new['cnt'] * 1000
cnt_daily_pred = cnt.shift(24)
cnt_weekly_pred = cnt.shift(24 * 7)

mae_daily = mean_absolute_error(cnt[24:], cnt_daily_pred[24:])
mae_weekly = mean_absolute_error(cnt[24*7:], cnt_weekly_pred[24*7:])

with open('mae_baseline.pkl', 'wb') as f:
    pickle.dump((mae_daily, mae_weekly), f)

mae_daily, mae_weekly


# In[9]:


cnt_train = df_new['cnt']['2011-01-01 00:00':'2012-06-30 23:00']
cnt_valid = df_new['cnt']['2012-07-01 00:00':]


# In[10]:


seq_len = 1 * 24
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_train.to_numpy(),
    targets=cnt_train[seq_len:],
    sequence_length=seq_len,
    batch_size=32,
    shuffle=True,
    seed=42
)
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_valid.to_numpy(),
    targets=cnt_valid[seq_len:],
    sequence_length=seq_len,
    batch_size=32
)


# In[11]:


model = tf.keras.Sequential([
    tf.keras.Input(shape=(seq_len,)),
    tf.keras.layers.Dense(1)
])


# In[12]:


model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    metrics=['mae']
)
history = model.fit(train_ds, validation_data=valid_ds, epochs=20)


# In[13]:


model.save('model_linear.keras')


# In[14]:


y_true = []
y_pred = []

for x_batch, y_batch in valid_ds:
    y_true.extend(y_batch.numpy())
    y_pred.extend(model.predict(x_batch).flatten())

y_true = [y * 1000 for y in y_true]
y_pred = [y * 1000 for y in y_pred]

mae_linear = mean_absolute_error(y_true, y_pred)

with open('mae_linear.pkl', 'wb') as f:
    pickle.dump((mae_linear,), f)

mae_linear


# In[15]:


model = tf.keras.Sequential([
    tf.keras.Input(shape=(seq_len, 1)),
    tf.keras.layers.LSTM(1)
])


# In[16]:


cnt_train = df_new['cnt']['2011-01-01 00:00':'2012-06-30 23:00'].to_numpy()
cnt_valid = df_new['cnt']['2012-07-01 00:00':].to_numpy()

X_train = tf.keras.utils.timeseries_dataset_from_array(
    cnt_train[:-seq_len].reshape(-1, 1),
    targets=cnt_train[seq_len:],
    sequence_length=seq_len,
    batch_size=32
)

X_valid = tf.keras.utils.timeseries_dataset_from_array(
    cnt_valid[:-seq_len].reshape(-1, 1),
    targets=cnt_valid[seq_len:],
    sequence_length=seq_len,
    batch_size=32
)


# In[17]:


model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    metrics=['mae']
)
history = model.fit(X_train, validation_data=X_valid, epochs=20)
model.save('model_rnn1.keras')


# In[18]:


y_true = []
y_pred = []

for x, y in X_valid:
    y_true.extend(y.numpy())
    y_pred.extend(model.predict(x).flatten())

y_true = [y * 1000 for y in y_true]
y_pred = [y * 1000 for y in y_pred]

mae_rnn1 = model.evaluate(X_valid)[1]

with open('mae_rnn1.pkl', 'wb') as f:
    pickle.dump((mae_rnn1,), f)

mae_rnn1


# In[19]:


model = tf.keras.Sequential([
    tf.keras.Input(shape=(seq_len, 1)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])


# In[20]:


model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    metrics=['mae']
)
history = model.fit(X_train, validation_data=X_valid, epochs=20)
model.save('model_rnn32.keras')


# In[21]:


y_true = []
y_pred = []

for x, y in X_valid:
    y_true.extend(y.numpy())
    y_pred.extend(model.predict(x).flatten())

y_true = [y * 1000 for y in y_true]
y_pred = [y * 1000 for y in y_pred]

mae_rnn32 = model.evaluate(X_valid)[1]

with open('mae_rnn32.pkl', 'wb') as f:
    pickle.dump((mae_rnn32,), f)

mae_rnn32


# In[22]:


model = tf.keras.Sequential([
    tf.keras.Input(shape=(seq_len, 1)),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])


# In[23]:


model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    metrics=['mae']
)

model.fit(X_train, validation_data=X_valid, epochs=20)

model.save('model_rnn_deep.keras')


# In[24]:


y_true, y_pred = [], []

for x, y in X_valid:
    y_true.extend(y.numpy())
    y_pred.extend(model.predict(x).flatten())

y_true = [v * 1000 for v in y_true]
y_pred = [v * 1000 for v in y_pred]

mae_rnn_deep = model.evaluate(X_valid)[1]

with open('mae_rnn_deep.pkl', 'wb') as f:
    pickle.dump((mae_rnn_deep,), f)

mae_rnn_deep


# In[32]:


seq_len = 24
features = ['cnt', 'atemp', 'weathersit', 'workingday']
df_features = df_new[features].copy()
#df_features['cnt'] *= 1000

X = df_features[features].copy()
y = df_features['cnt'].copy()

X_train = X['2011-01-01 00:00':'2012-06-30 23:00'].to_numpy()
X_valid = X['2012-07-01 00:00':].to_numpy()

y_train = y['2011-01-01 00:00':'2012-06-30 23:00'].to_numpy()
y_valid = y['2012-07-01 00:00':].to_numpy()


# In[33]:


train_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=X_train,
    targets=y_train[seq_len:],
    sequence_length=seq_len,
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    data=X_valid,
    targets=y_valid[seq_len:],
    sequence_length=seq_len,
    batch_size=32
)


# In[34]:


model = tf.keras.Sequential([
    tf.keras.Input(shape=(seq_len, 4)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])


# In[37]:


model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
    metrics=['mae']
)

model.fit(train_ds, validation_data=valid_ds, epochs=20)
model.save('model_rnn_mv.keras')


# In[38]:


y_true, y_pred = [], []

for x_batch, y_batch in valid_ds:
    y_true.extend(y_batch.numpy())
    y_pred.extend(model.predict(x_batch).flatten())

mae_rnn_mv = model.evaluate(valid_ds)[1]*1e3

with open('mae_rnn_mv.pkl', 'wb') as f:
    pickle.dump((mae_rnn_mv,), f)

mae_rnn_mv

