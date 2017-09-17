import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

# creating a dataset


# extracting pics of gajendra
path = "gajsdataset"
image_path = [os.path.join(path, f)for f in os.listdir(path)]
gajsdata = []
label = []
for image in image_path:
    read = Image.open(image)
    read1 = np.array(read)
    gajsdata.append(read1)
    label.append(1)

# Creaing my own dataset

path1 = '/home/abhishek/PycharmProjects/untitled1/traindata'
image_path = [os.path.join(path, f)for f in os.listdir(path)]
mydata = []

for image in image_path:
    read = Image.open(image)
    read1 = np.array(read)
    mydata.append(read1)
    label.append(2)

train_x = np.column_stack((gajsdata,mydata))
print(train_x)
print(label)
gajsdata.extend(mydata)
print(len(gajsdata))

data = pd.DataFrame({'pixels':gajsdata, 'label':label})

label = data['label']
data = data.drop('label', axis=1)
print(type(data['pixels'].loc[0]))
# done with creating a dataset

# now time for deep learning

# Setting up hyper-parameters
n_inpus = 192
n_classes = 2
training_epoch = 1000
learning_rate = 0.001

# setting up placeholder
x = tf.placeholder(tf.float32, [None, n_inpus])
y_ = tf.placeholder(tf.float32, [None, n_classes])

hidden_layer1 = 60
hidden_layer2 = 60
hidden_layer3 = 60

# now setting up variables
w = {'w1': tf.Variable(tf.truncated_normal([n_inpus, hidden_layer1])),
     'w2': tf.Variable(tf.truncated_normal([hidden_layer1, hidden_layer2])),
     'w3': tf.Variable(tf.truncated_normal([hidden_layer2, hidden_layer3])),
     'out': tf.Variable(tf.truncated_normal([hidden_layer3, n_classes]))}
b = {'b1': tf.Variable(tf.zeros([hidden_layer1])),
     'b2': tf.Variable(tf.zeros([hidden_layer2])),
     'b3': tf.Variable(tf.zeros([hidden_layer3])),
     'out': tf.Variable(tf.zeros([n_classes]))}

# now creating our model

layer1 = tf.add(tf.matmul(x, w['w1']), b['b1'])
layer1_relu = tf.nn.relu(layer1)

# 2nd layer
layer2 = tf.add(tf.matmul(layer1_relu, w['w2']), b['b2'])
layer2_relu = tf.nn.relu(layer2)

# 3rd layer
layer3 = tf.add(tf.matmul(layer2_relu, w['w3']), b['b3'])
layer3_relu = tf.nn.relu(layer3)

# output layer
output = tf.matmul(layer3_relu, w['out']) + b['out']

y = output
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initializing the variables
init = tf.global_variables_initializer()

# creating a session
sess = tf.Session()
sess.run(init)

a = data['pixels'].loc[1][1]
df = pd.DataFrame()

#for epoch in range(training_epoch):
 #   sess.run(training_step, {x: train_x, y_: label})



















