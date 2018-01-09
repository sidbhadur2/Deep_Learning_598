import tensorflow as tf

import numpy as np
import os
import time

from multiprocessing import Pool

from helperFunctions import getUCF101
from helperFunctions import loadSequence
import h5py

data_directory = '/u/training/tra044/scratch/HW5/HW5_Action_Recognition/'
class_list, train, test = getUCF101(base_directory = data_directory)

sequence_length = 15
sequence_length_test = 200
batch_size = 1
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
saver.restore(sess,tf.train.latest_checkpoint('/u/training/tra044/scratch/HW5/HW5_Action_Recognition/rnn_model/'))

prediction_directory = 'UCF-101-predictions-2/'
if not os.path.exists(prediction_directory):
    os.makedirs(prediction_directory)
for label in class_list:
    if not os.path.exists(prediction_directory+label+'/'):
        os.makedirs(prediction_directory+label+'/')


acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((num_classes,num_classes),dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))


for i in range(0, len(test[0])-batch_size,batch_size):
    t1 = time.time()

    index = random_indices[i]
    temp = (test[0][index], sequence_length_test, False)
    #data = pool_threads.map(loadSequence,video_list)

    data = loadSequence(temp)
    data = np.asarray(data, dtype = np.float32)

    print("DATA SHAPE: ")
    print(data.shape)  
    next_batch = 0

    check_dim = data.shape[0]
    if (check_dim == 0):
        print ("ZERO LENGTH FILE")
        continue
    #for video in data:
    #    if video.size==0: # there was an exception, skip this
    #        next_batch = 1
    #if(next_batch==1):
    #    continue

    first_dim = data.shape[0]
    second_dim = data.shape[1]
    data = np.reshape(data,(first_dim,1,second_dim))

    curr_pred = sess.run(probabilities_test,feed_dict={X_sequence_test: data, keep_prob:1.0})
    curr_pred = np.asarray(curr_pred)
    curr_pred = np.reshape(curr_pred, (200, 101))

    filename = ''    
    
    filename = filename + test[0][index]
    print (filename)
    filename = filename.replace('.avi','.hdf5')

    filename = filename.replace('UCF-101','UCF-101-hdf5')
    print (filename)
    filename = filename.replace(data_directory+'UCF-101-hdf5/', prediction_directory)

    if(not os.path.isfile(filename)):
        with h5py.File(filename,'w') as h:
            h.create_dataset('predictions',data=curr_pred)

    label = test[1][index]
    log_prob = np.log(curr_pred)
    pred = np.mean(log_prob, axis = 0)
    argsort_pred = np.argsort(-pred)[0:10]

    confusion_matrix[label,argsort_pred[0]] += 1
    if(label==argsort_pred[0]):
        acc_top1 += 1.0
    if(np.any(argsort_pred[0:5]==label)):
        acc_top5 += 1.0
    if(np.any(argsort_pred[:]==label)):
        acc_top10 += 1.0

    print('i:%d t:%f (%f,%f,%f)' 
          % (i,time.time()-t1,acc_top1/(i+1),acc_top5/(i+1), acc_top10/(i+1)))

number_of_examples = np.sum(confusion_matrix,axis=1)

for i in range(num_classes):
    confusion_matrix[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]


fig_dir = '/u/training/tra044/scratch/HW5/HW5_Action_Recognition/'
#fig_name = str(1) + ".png"
#fig_loc = fig_dir + fig_name
file_name = str(2) + ".csv"
file_loc = fig_dir + file_name

np.savetxt(file_loc, confusion_matrix, delimiter=",")

#plt.pcolormesh(confusion_matrix, cmap='RdBu_r')
#plt.savefig(fig_loc)


for i in range(num_classes):
    print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])