import tensorflow as tf

import numpy as np
import os
import time

from multiprocessing import Pool

from helperFunctions import getUCF101
from helperFunctions import loadSequence
import h5py

data_directory = ''
class_list, train, test = getUCF101(base_directory = data_directory)

sequence_length = 15
sequence_length_test = 200
batch_size = 64
num_of_features = 1024
num_classes = 101

W_output = tf.get_variable('W_output', 
                            [num_of_features, num_classes], 
                            initializer=tf.contrib.layers.xavier_initializer())
B_output = tf.get_variable('B_output', 
                            [num_classes], 
                            initializer=tf.constant_initializer())

X_sequence = tf.placeholder(tf.float32, 
                            [sequence_length, batch_size, num_of_features])
X_sequence_test = tf.placeholder(tf.float32,
                            [sequence_length_test, batch_size, num_of_features])
y = tf.placeholder(name='label',dtype=tf.float32,shape=[batch_size,num_classes])
keep_prob = tf.placeholder(tf.float32 ,shape=())

lstm = tf.contrib.rnn.BasicLSTMCell(num_of_features)
hidden_state = tf.zeros([batch_size, num_of_features])
current_state = tf.zeros([batch_size, num_of_features])
state = hidden_state, current_state
state_test = hidden_state, current_state

probabilities = []
probabilities_test = []
loss = 0.0
with tf.variable_scope('network'):
    for i in range(sequence_length):
        if(i>0):
            tf.get_variable_scope().reuse_variables()
        output, state = lstm(X_sequence[i,:,:], state)
        output = tf.nn.dropout(output,keep_prob=keep_prob)
        logits = tf.matmul(output, W_output) + B_output

        probabilities.append(tf.nn.softmax(logits))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        logits = logits, labels = y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        loss += cross_entropy_mean/float(sequence_length)

    for i in range(sequence_length_test):
        tf.get_variable_scope().reuse_variables()
        output_test, state_test = lstm(X_sequence_test[i,:,:], state_test)
        output_test = tf.nn.dropout(output_test,keep_prob=keep_prob)
        logits_test = tf.matmul(output_test, W_output) + B_output

        probabilities_test.append(tf.nn.softmax(logits_test))


tvar = tf.trainable_variables()
all_vars = tf.global_variables()

opt = tf.train.MomentumOptimizer(0.01, 0.9)
grads = opt.compute_gradients(loss)
train_op = opt.apply_gradients(grads)

sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver(all_vars)

pool_threads = Pool(10,maxtasksperchild=200)


for epoch in range(0,100):

    # ###### TEST
    if(epoch%5==0):
        random_indices = np.random.permutation(len(test[0]))
        prob_test = 0.0
        count = 0
        is_train = False
        for i in range(0, 500-batch_size,batch_size):
            video_list = [(test[0][k],sequence_length_test,is_train)
                            for k in random_indices[i:(batch_size+i)]]
            data = pool_threads.map(loadSequence,video_list)

            next_batch = 0
            for video in data:
                if video.size==0: # there was an exception, skip this
                    next_batch = 1
            if(next_batch==1):
                continue

            data = np.asarray(data,dtype=np.float32)
            data = data.transpose(1,0,2)

            labels_batch = np.asarray(test[1][random_indices[i:(batch_size+i)]])
            y_batch = np.zeros((batch_size,num_classes),dtype=np.float32)
            y_batch[np.arange(batch_size),labels_batch] = 1

            prob = sess.run(probabilities_test,
                   feed_dict={X_sequence_test: data, keep_prob:1.0})

            prob = np.asarray(prob)
            log_prob = np.log(prob)
            log_prob = np.mean(log_prob,axis=0)
            pred = np.argmax(log_prob,axis=1)

            prob_p = np.sum(pred==labels_batch)/float(batch_size)

            prob_test += prob_p
            count += 1

        print('TEST: %f' % (prob_test/count))

    ###### TRAIN
    random_indices = np.random.permutation(len(train[0]))
    count = 0
    is_train = True
    for i in range(0, len(train[0])-batch_size,batch_size):

        t1 = time.time()

        video_list = [(train[0][k],sequence_length,is_train)
                     for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadSequence,video_list)

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this
                next_batch = 1
        if(next_batch==1):
            continue

        data = np.asarray(data,dtype=np.float32)
        data = data.transpose(1,0,2)

        t_data_load = time.time()-t1

        labels_batch = np.asarray(train[1][random_indices[i:(batch_size+i)]])

        y_batch = np.zeros((batch_size,num_classes),dtype=np.float32)
        y_batch[np.arange(batch_size),labels_batch] = 1

        _, loss_p, prob = sess.run([train_op,loss,probabilities],
                          feed_dict={X_sequence: data, y: y_batch, keep_prob:1.0})

        prob = np.asarray(prob)
        log_prob = np.log(prob)
        log_prob = np.mean(log_prob,axis=0)
        pred = np.argmax(log_prob,axis=1)

        prob_p = np.sum(pred==labels_batch)/float(batch_size)
        t_train = time.time() - t1 - t_data_load

        count += 1
        if(count%10 == 0):
            print('epoch: %d i: %d t_load: %f t_train: %f loss: %f acc: %f'
                  % (epoch,i,t_data_load,t_train,loss_p,prob_p))

if not os.path.exists('rnn_model/'):
    os.makedirs('rnn_model/')
all_vars = tf.global_variables()
saver = tf.train.Saver(all_vars)
saver.save(sess,'rnn_model/model')

pool_threads.terminate()
pool_threads.close()

