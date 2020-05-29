
# coding: utf-8

# In[ ]:


import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import random


# In[80]:


sent_one_data = []
sent_two_data = []
label_data = []


sep_token_num = 39999

data = open('/content/drive/My Drive/teck_task/train.csv')
lines = data.readlines()
lines = lines[1:]
random.shuffle(lines)

train_data = []
train_label = []
vocabulary_set = set()

valid_data = []
valid_label = []


for i, line in enumerate(lines):
    line = line.replace('\n', '')
    tokens = line.split(',')

    sent1 = tokens[1]
    sent2 = tokens[2]
    label = int(tokens[3])

    sent_data = sent1 + ' ' + str(sep_token_num) + ' ' + sent2

    for word in sent_data.split(' '):
        vocabulary_set.add(word)
    
    if i > len(lines) * 0.9:
        valid_data.append(sent_data)
        valid_label.append(label)
    else:
        train_data.append(sent_data)
        train_label.append(label)

data.close()

print(len(train_data))
print(len(valid_data))


# In[ ]:


encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


# In[82]:


print(train_data[0])
tokenized_string = encoder.encode(train_data[0])
for ts in tokenized_string:
    print ('{}: {}'.format(encoder.decode([ts]), ts))


# In[ ]:


lines_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(np.array(train_data), tf.string),
            tf.cast(np.array(train_label), tf.float32)           
        )
    )
)


# In[ ]:


lines_valid_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(np.array(valid_data), tf.string),
            tf.cast(np.array(valid_label), tf.float32)           
        )
    )
)


# In[ ]:


BUFFER_SIZE = 20000
BATCH_SIZE = 1024
MAX_LENGTH = 64


# In[ ]:


def encode(sent, label):
    sent = [encoder.vocab_size] + encoder.encode(sent.numpy()) + [encoder.vocab_size+1]
    return sent, [label]


# In[ ]:


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)


# In[ ]:


def tf_encode(sent,label):
    return tf.py_function(encode, [sent,label], [tf.int64, tf.float32])


# In[ ]:


train_dataset = lines_dataset.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


# In[ ]:


valid_dataset = lines_valid_dataset.map(tf_encode)
valid_dataset = valid_dataset.cache()
valid_dataset = valid_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)


# In[91]:


train_dataset


# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 128),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5), merge_mode='mul'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.5)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[95]:


model.summary()


# In[96]:


model.fit(train_dataset, epochs=20, validation_data=valid_dataset)


# In[ ]:


model.save('/content/drive/My Drive/teck_task/model1_bilstm')


# In[ ]:


def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec


# In[ ]:


def sample_predict(sentence, pad):
    encoded_sample_pred_text = encoder.encode(sentence)

    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.int64)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0), steps=1)

    return int(round(predictions.item(0)))


# In[ ]:


test_data = []
test_label = []

correct_num = 0
total_cnt = 0

for i, line in enumerate(lines[35001:]):
    if (i+1) % 500 == 0:
        print(i, '---->', correct_num/total_cnt)
    
    total_cnt += 1
    line = line.replace('\n', '')
    tokens = line.split(',')

    id_ = tokens[0]
    sent1 = tokens[1]
    sent2 = tokens[2]
    label = int(tokens[3])

    sent_data = sent1 + ' ' + str(sep_token_num) + ' ' + sent2
    result = sample_predict(sent_data, True)
    if result == label:
        correct_num += 1


# In[ ]:


print(correct_num/total_cnt)


# In[ ]:


import csv    
f = open('/content/drive/My Drive/teck_task/test_result.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(['id', 'label'])


data = open('/content/drive/My Drive/teck_task/test.csv')

lines = data.readlines()

test_data = []
test_label = []

for i, line in enumerate(lines[1:]):
    line = line.replace('\n', '')
    tokens = line.split(',')

    id_ = tokens[0]
    sent1 = tokens[1]
    sent2 = tokens[2]

    sent_data = sent1 + ' ' + str(sep_token_num) + ' ' + sent2
    result = sample_predict(sent_data, True)
    print(i, sent_data, result)
    wr.writerow([id_, result])

f.close()

