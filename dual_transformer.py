
# coding: utf-8

# In[2]:


import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import time

import math

import random

tf.compat.v1.enable_eager_execution()


# Preprocessing

# In[ ]:


sent1_data = []
sent2_data = []
label_data = []
vocabulary_set = set()

sep_token_num = 39999

data = open('/content/drive/My Drive/teck_task/train.csv')
lines = data.readlines()
lines = lines[1:]
random.shuffle(lines)

sent_data = []
train_label = []

for line in lines[:35000]:
    line = line.replace('\n', '')
    tokens = line.split(',')

    token_ids = []
    sent1 = tokens[1]

    sent2 = tokens[2]
    sent1_data.append(sent1)
    sent2_data.append(sent2)

    label = int(tokens[3])

    sent_string_data = sent1 + ' ' + str(sep_token_num) + ' ' + sent2
    for word in sent_string_data.split(' '):
        vocabulary_set.add(word)
    
    train_label.append(label)


# In[4]:


print(sent1_data[0:2])
print(sent2_data[0:2])
print(train_label[0:2])


# In[ ]:


encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


# In[6]:


print(sent1_data[0])
tokenized_string = encoder.encode(sent1_data[0])
for ts in tokenized_string:
    print ('{}: {}'.format(encoder.decode([ts]), ts))


# In[7]:


sep_token_vocab_num = encoder.encode(str(sep_token_num))[0]
print('sep_token_vocab_num:', sep_token_vocab_num)


# In[ ]:


BUFFER_SIZE = 20000
BATCH_SIZE = 128
MAX_LENGTH = 64


# In[ ]:


def encode(sent1, sent2, label):
    sent1 = [encoder.vocab_size] + encoder.encode(sent1.numpy()) + [encoder.vocab_size+1]
    if len(sent1) < MAX_LENGTH:
        sent1 += ([0] * (MAX_LENGTH-len(sent1)))
    sent2 = [encoder.vocab_size] + encoder.encode(sent2.numpy()) + [encoder.vocab_size+1]
    if len(sent2) < MAX_LENGTH:
        sent2 += ([0] * (MAX_LENGTH-len(sent2)))    
    return sent1, sent2, [label]


# In[ ]:


def filter_max_length(x, y, z, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)


# In[ ]:


def tf_encode(sent1, sent2, label):
    return tf.py_function(encode, [sent1, sent2, label], [tf.float64, tf.float64, tf.float64])


# In[ ]:


lines_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(np.array(sent1_data), tf.string),
            tf.cast(np.array(sent2_data), tf.string),
            tf.cast(np.array(train_label), tf.float64)
        )
    )
)


# In[ ]:


train_dataset = lines_dataset.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


# In[14]:


train_dataset


# In[ ]:


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                              np.arange(d_model)[np.newaxis, :],
                              d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# In[ ]:


def create_masks(inp):
    enc_padding_mask = create_padding_mask(inp)
    return enc_padding_mask


# In[ ]:


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


# In[ ]:


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


# In[ ]:


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


# In[ ]:


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
    
        assert d_model % self.num_heads == 0
    
        self.depth = d_model // self.num_heads
    
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
    
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
    
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
    
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
    
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)
        
        return output, attention_weights


# In[ ]:


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(d_model)])


# In[ ]:


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
    
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
    
        return out2


# In[ ]:


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x


# In[ ]:


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                   label_num, rate=0.1):
        super(Transformer, self).__init__()

        self.sent1_encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, rate)
        self.sent2_encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, rate)
                
        self.cocat_layer = tf.keras.layers.Concatenate()

        self.dense1 = tf.keras.layers.Dense(d_model, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inp1, inp2, training, sent1_enc_padding_mask, sent2_enc_padding_mask):
        sent1_enc_output = self.sent1_encoder(inp1, training, sent1_enc_padding_mask)
        sent2_enc_output = self.sent2_encoder(inp2, training, sent2_enc_padding_mask)
        cocat_layer_output = self.cocat_layer([sent1_enc_output, sent2_enc_output])

        enc_output = self.dense1(cocat_layer_output[:,0])
        enc_output = self.dropout1(enc_output, training=training)
        final_output = self.final_layer(enc_output)

        return final_output


# In[ ]:


vocab_size = encoder.vocab_size + 2
num_layers = 1
d_model = 1024
dff = 1024
num_heads = 8

input_vocab_size = vocab_size
dropout_rate = 0.5


# In[ ]:


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
    
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# In[ ]:


learning_rate = CustomSchedule(d_model)
# learning_rate = 0.01

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)


# In[ ]:


loss_object = tf.keras.losses.BinaryCrossentropy()


# In[ ]:


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)


# In[ ]:


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(
    name='train_accuracy')


# In[ ]:


transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, 2, dropout_rate)


# In[ ]:


checkpoint_path = "/content/drive/My Drive/teck_task/model_output_concat_"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint is restored')


# In[ ]:


@tf.function
def train_step(inp1, intp2, tar):
    sent1_enc_padding_mask = create_masks(inp1)
    sent2_enc_padding_mask = create_masks(inp2)
    with tf.GradientTape() as tape:
        predictions = transformer(inp1, inp2, True, sent1_enc_padding_mask, sent2_enc_padding_mask)
        loss = loss_function(tar, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar, predictions)


# In[ ]:


EPOCHS = 1


# In[ ]:


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp1, inp2, tar)) in enumerate(train_dataset):
        train_step(inp1, inp2, tar)

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
              epoch + 1, batch, train_loss.result(), train_accuracy.result()))
    
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))
        print ('Epoch {} Train_Loss {:.4f} Train_Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result()))
        print ('Time taken for 5 epoch: {} secs\n'.format(time.time() - start))


# In[ ]:


def encode_for_inference(sent1, sent2):
    sent1 = [encoder.vocab_size] + encoder.encode(sent1) + [encoder.vocab_size+1]
    if len(sent1) < MAX_LENGTH:
        sent1 += ([0] * (MAX_LENGTH-len(sent1)))
    sent2 = [encoder.vocab_size] + encoder.encode(sent2) + [encoder.vocab_size+1]
    if len(sent2) < MAX_LENGTH:
        sent2 += ([0] * (MAX_LENGTH-len(sent2)))    
    return sent1, sent2


# In[ ]:


def evaluate(sent1_input, sent2_input):
    
    sent1_input, sent2_input = encode_for_inference(sent1_input, sent2_input)

    sent1_encoder_input = tf.expand_dims(sent1_input, 0)
    sent2_encoder_input = tf.expand_dims(sent2_input, 0)
    output = tf.expand_dims(2, 0)  

    sent1_enc_padding_mask = create_masks(sent1_encoder_input)
    sent2_enc_padding_mask = create_masks(sent2_encoder_input)
    predictions = transformer(sent1_encoder_input, sent2_encoder_input, False, sent1_enc_padding_mask, sent2_enc_padding_mask)
    predictions = tf.round(predictions)

    return int(tf.get_static_value(predictions))


# In[ ]:


print(evaluate('17244 28497 16263',  '17244 3992 6582 19891 16992 19969 1044'))


# In[ ]:


# dev set test
# data = open('/content/drive/My Drive/teck_task/train.csv')

# lines = data.readlines()

test_data = []
test_label = []

correct_num = 0
total_cnt = 0

for i, line in enumerate(lines[35001:]):
    if (i+1) % 500 == 0:
        print(i, '--->', correct_num/total_cnt)
    total_cnt += 1
    line = line.replace('\n', '')
    tokens = line.split(',')

    id_ = tokens[0]
    sent1 = tokens[1]
    sent2 = tokens[2]
    label = int(tokens[3])

    result = evaluate(sent1, sent2)
    if result == label:
        correct_num += 1

print('final accuracy:', correct_num/total_cnt)


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
    result = evaluate(sent_data)
    wr.writerow([id_, result])

f.close()

