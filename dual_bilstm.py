
# coding: utf-8

# In[ ]:


import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import random

tf.compat.v1.enable_eager_execution()


# In[ ]:


sep_token_num = 39999

data = open('/content/drive/My Drive/teck_task/train.csv')
lines = data.readlines()

vocabulary_set = set()
max_len = 0

total_data_cnt = len(lines)


for line in lines[1:]:
    line = line.replace('\n', '')
    tokens = line.split(',')

    sent1 = tokens[1]
    sent2 = tokens[2]
    
    for word in sent1.split(' '):
        vocabulary_set.add(word)
    for word in sent2.split(' '):
        vocabulary_set.add(word)
    if len(sent1) > max_len:
        max_len = len(sent1)
    if len(sent2) > max_len:
        max_len = len(sent2)
    
data.close()


# In[6]:


print(len(vocabulary_set))


# In[ ]:


encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


# In[ ]:


BUFFER_SIZE = 20000
BATCH_SIZE = 128
MAX_LENGTH = 64


# In[ ]:


def encode(sent):
    sent = encoder.encode(sent)
    return sent


# In[ ]:


sent_one_data = []
sent_two_data = []


sep_token_num = 39999

data = open('/content/drive/My Drive/teck_task/train.csv')
lines = data.readlines()
lines = lines[1:]
random.shuffle(lines)

sent1_train_data = []
sent2_train_data = []
train_label = []


pos_num = 0
neg_num = 0


for line in lines:
    line = line.replace('\n', '')
    tokens = line.split(',')
    sent1 = tokens[1]
    sent2 = tokens[2]
    label = int(tokens[3])
    encoded_sent1 = encode(sent1)
    if len(encoded_sent1) < 100:
        encoded_sent1.extend([0] * (100-len(encoded_sent1)))
    encoded_sent2 = encode(sent2)
    if len(encoded_sent2) < 100:
        encoded_sent2.extend([0] * (100-len(encoded_sent2)))
    sent1_train_data.append(encoded_sent1)
    sent2_train_data.append(encoded_sent2)
    train_label.append(label)
data.close()


# In[ ]:


sent1 = tf.keras.Input(shape=(None,), name='sent1')
sent1_embedding = tf.keras.layers.Embedding(encoder.vocab_size, 32)(sent1)
sent1_embedding = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(18,return_sequences=True, dropout=0.5))(sent1_embedding)


# In[ ]:


sent2 = tf.keras.Input(shape=(None,), name='sent2')
sent2_embedding = tf.keras.layers.Embedding(encoder.vocab_size, 32)(sent2)
sent2_embedding = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(18,return_sequences=True, dropout=0.5))(sent2_embedding)


# In[ ]:


cocat_layer = tf.keras.layers.concatenate([sent1_embedding, sent2_embedding])
lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, dropout=0.5))(cocat_layer)
# x = tf.keras.layers.Dense(8, activation='relu')(cocat_layer)
# x = tf.keras.layers.Dropout(0.5)(x)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(lstm_layer)


# In[ ]:


model = tf.keras.Model(inputs=[sent1, sent2], outputs=output_layer)


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[35]:


model.summary()


# In[36]:


model.fit({'sent1':np.array(sent1_train_data), 'sent2':np.array(sent2_train_data)}, {'output':np.array(train_label)},epochs=20,batch_size=512, shuffle=True, validation_split=0.1)

