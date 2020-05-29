
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
import random


# In[ ]:


data = open('/content/drive/My Drive/teck_task/train.csv')

lines = data.readlines()
lines = lines[1:]
random.shuffle(lines)

train_data = []
train_label = []
vocabulary_set = set()

valid_data = []
valid_label = []

sep_token_num = 39999

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


# In[ ]:


vectorizer = TfidfVectorizer(min_df = 1,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)


# In[ ]:


train_vectors = vectorizer.fit_transform(train_data)
valid_vectors = vectorizer.transform(valid_data)


# In[ ]:


import time
from sklearn import svm
from sklearn.metrics import classification_report


# In[ ]:


classifier_linear = svm.SVC(kernel='linear')

t0 = time.time()
classifier_linear.fit(train_vectors, train_label)
t1 = time.time()
prediction_linear = classifier_linear.predict(valid_vectors)
t2 = time.time()


# In[8]:


time_linear_train = t1-t0
time_linear_predict = t2-t1
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))

report = classification_report(valid_label, prediction_linear, output_dict=True)


# In[9]:


print('positive: ', report['1'])
print('negative: ', report['0'])


# In[11]:


total_pos_cnt = 0
total_neg_cnt = 0
Y = valid_label
Y_pred = prediction_linear
for i in Y:
    if i == 1:
        total_pos_cnt += 1
    else:
        total_neg_cnt += 1

print(total_pos_cnt, total_neg_cnt)
pos_cnt = 0
neg_cnt = 0

for i in range(0, len(Y)):
    if Y[i] == Y_pred[i]:
        if Y[i] == 1:
            pos_cnt += 1
        else:
            neg_cnt += 1

print(pos_cnt, neg_cnt)
print('True accuracy:', pos_cnt/total_pos_cnt)
print('False accuracy:', neg_cnt/total_neg_cnt)
print('Total accuracy:', (pos_cnt+neg_cnt)/(total_pos_cnt+total_neg_cnt))

