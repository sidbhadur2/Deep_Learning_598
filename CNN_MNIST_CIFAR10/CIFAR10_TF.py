import numpy as np
import tensorflow as tf
import time


#learning rate
LR = .0001
batch_size = 100
training_epochs = 1000
num_batches = 50
num_channels1 = 16*4
num_channels2 = 32*4
num_channels3 = 64*4
num_H1 = 10
num_H2 = 100
keep_prob0 = .8

#load data 
import h5py
CIFAR10_data = h5py.File('CIFAR10.hdf5', 'r')
x_train0 = np.float32(CIFAR10_data['X_train'][:] )
y_train = np.int32(np.array(CIFAR10_data['Y_train'][:]))
x_test0 = np.float32(CIFAR10_data['X_test'][:] )
y_test = np.int32( np.array(CIFAR10_data['Y_test'][:]  ) )
CIFAR10_data.close()
L_Y_train = len(y_train)

#re-shape x data to match TensorFlow format for convolutions
x_train = np.zeros(  (len(y_train), 32, 32, 3)   )
x_test = np.zeros(  (len(y_test), 32, 32, 3)  )
x_train[:,:,:,0] = x_train0[:,0,:,:]                   
x_train[:,:,:,1] = x_train0[:,1,:,:] 
x_train[:,:,:,2] = x_train0[:,2,:,:]  
x_test[:,:,:,0] = x_test0[:,0,:,:]                   
x_test[:,:,:,1] = x_test0[:,1,:,:] 
x_test[:,:,:,2] = x_test0[:,2,:,:]                    


print "Learning rate: %f" % LR
print "Batch size: %d" % batch_size
print "Dropout probability: %f" % keep_prob0
print "Number of units in H1: %d" % num_H1
print "Number of units in H2: %d" % num_H2



#PLACEHOLDERS FOR DATA
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="x-input")
#x = tf.reshape(x, [-1,32,32,3])
# target 10 output classes
y = tf.placeholder(tf.int64, shape=[None], name="y-input")
keep_prob = tf.placeholder(tf.float32)


#PARAMATERS

#H,W, # of input channels, # of output channels
W1 = tf.Variable(tf.truncated_normal([3,3,3,num_channels1], stddev=1.0/np.sqrt(3*3*3)))
W2 = tf.Variable(tf.truncated_normal([3,3,num_channels1,num_channels1], stddev=1.0/np.sqrt(3*3*num_channels1) ))

W3 = tf.Variable(tf.truncated_normal([3,3,num_channels1,num_channels2], stddev= 1.0/np.sqrt(num_channels1*3*3)))
W4 = tf.Variable(tf.truncated_normal([3,3,num_channels2,num_channels2], stddev=1.0/np.sqrt(3*3*num_channels2)))

W5 = tf.Variable(tf.truncated_normal([3,3,num_channels2,num_channels3], stddev=1.0/np.sqrt(3*3*num_channels2) ))
W6 = tf.Variable(tf.truncated_normal([3,3,num_channels3,num_channels3], stddev=1.0/np.sqrt(3*3*num_channels3) ))
W7 = tf.Variable(tf.truncated_normal([3,3,num_channels3,num_channels3], stddev=1.0/np.sqrt(3*3*num_channels3) ))
W8 = tf.Variable(tf.truncated_normal([3,3,num_channels3,num_channels3], stddev=1.0/np.sqrt(3*3*num_channels3) ))


    
W_fc1 = tf.Variable( tf.truncated_normal([4*4*num_channels3, num_H1], stddev= 1.0/np.sqrt(4*4*num_channels3) ) )
    #     W_fc2 = tf.Variable( tf.truncated_normal([num_H2,10], stddev= 1.0/np.sqrt(num_H1) ) )   
W_fc2 = tf.Variable( tf.truncated_normal([num_H1,num_H2], stddev= 1.0/np.sqrt(num_H1) ) )
W_fc3 = tf.Variable( tf.truncated_normal([num_H2,10], stddev= 1.0/np.sqrt(num_H2) ) )


#NETWORK
C1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides = [1,1,1,1], padding = "SAME") )
C2 = tf.nn.relu(tf.nn.conv2d(C1, W2, strides = [1,1,1,1], padding = "SAME") )
P1 = tf.nn.max_pool(C2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
P1_drop = tf.nn.dropout(P1, keep_prob)

C3 = tf.nn.relu(tf.nn.conv2d(P1_drop, W3, strides = [1,1,1,1], padding = "SAME") )
C4 = tf.nn.relu(tf.nn.conv2d(C3, W4, strides = [1,1,1,1], padding = "SAME") )
P2 = tf.nn.max_pool(C4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
P2_drop = tf.nn.dropout(P2, keep_prob)

C5  = tf.nn.relu(tf.nn.conv2d(P2_drop, W5, strides = [1,1,1,1], padding = "SAME") )
C6 = tf.nn.relu(tf.nn.conv2d(C5, W6, strides = [1,1,1,1], padding = "SAME") )
C7 = tf.nn.relu(tf.nn.conv2d(C6, W7, strides = [1,1,1,1], padding = "SAME") )
C8 = tf.nn.relu(tf.nn.conv2d(C7, W8, strides = [1,1,1,1], padding = "SAME") )
P3 =  tf.nn.max_pool(C8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
P3_drop = tf.nn.dropout(P3, keep_prob)

    #out size = [32,32
    #H1 = tf.nn.relu()
P3_flat =  tf.reshape(P3_drop, [-1, 4*4*num_channels3])
H1 = tf.nn.relu( tf.matmul(P3_flat, W_fc1))
H1_drop = tf.nn.dropout(H1, keep_prob)
H2 = tf.nn.relu( tf.matmul(H1_drop, W_fc2))
H2_drop = tf.nn.dropout(H2, keep_prob)
u = tf.matmul(H2_drop, W_fc3)


#cross-entropy loss
cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=u))

#calculate accuracy on the current batch
correct_prediction = tf.equal(tf.argmax(u,1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#type of optimizer and learning rate
opt = tf.train.AdamOptimizer(LR)

   
train_op = opt.minimize(cross_entropy_loss)
    
    
#initialize session
init_op = tf.initialize_all_variables()
session = tf.Session()
session.run(init_op)

    
#training
    
    
time1 = time.time()
for epochs in range(training_epochs):
        #randomly scramble data
    I_permutation = np.random.permutation(L_Y_train)
    x_train = x_train[I_permutation,:]
    y_train = y_train[I_permutation]           
    for i in range(num_batches):         
        batch_x = x_train[i*batch_size:(i+1)*batch_size,:]
        batch_y = y_train[i*batch_size:(i+1)*batch_size]
        _, cost, accuracy_np = session.run( [train_op, cross_entropy_loss, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: keep_prob0})
        #test accuracy
    time2 = time.time()      
    time_elapsed = time2 - time1        
    accuracy_total_test = 0.0
    for j in range(0, 10000, batch_size):
        accuracy_np = session.run( accuracy, feed_dict={x: x_test[j:j+batch_size,:], y: y_test[j:j+batch_size], keep_prob: 1.0})
        accuracy_total_test += accuracy_np*100.0
    accuracy_percent = accuracy_total_test/np.float(10000.0/batch_size)  
    print "epoch %d, time %f: test accuracy = %f" % (epochs,time_elapsed, accuracy_percent)
   





