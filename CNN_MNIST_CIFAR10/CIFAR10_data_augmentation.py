import tensorflow as tf

def conv(x, w, b, stride, name):
    with tf.variable_scope('conv'):
        return tf.nn.conv2d(x,
                           filter=w,
                           strides=[1, stride, stride, 1],
                           padding='SAME',
                           name=name) + b

######## after 30k iterations (batch_size=64)
# with data augmentation (flip, brightness, contrast) ~81.0%
# without data augmentation 69.6%
def cifar10_conv(X, keep_prob, reuse=False):
    with tf.variable_scope('cifar10_conv'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        batch_size = tf.shape(X)[0]
        
        #K = 64
        #M = 128
        #N = 256

    num_channels1 = 16*4
    num_channels2 = 32*4
    num_channels3 = 63*3
    num_H1 = 10
    num_H2 = 100

    W1 = tf.Variable(tf.truncated_normal([3,3,3,num_channels1], stddev=1.0/np.sqrt(3*3*3)))
    W2 = tf.Variable(tf.truncated_normal([3,3,num_channels1,num_channels1], stddev=1.0/np.sqrt(3*3*num_channels1) ))

    W3 = tf.Variable(tf.truncated_normal([3,3,num_channels1,num_channels2], stddev= 1.0/np.sqrt(num_channels1*3*3)))
    W4 = tf.Variable(tf.truncated_normal([3,3,num_channels2,num_channels2], stddev=1.0/np.sqrt(3*3*num_channels2)))

    W5 = tf.Variable(tf.truncated_normal([3,3,num_channels2,num_channels3], stddev=1.0/np.sqrt(3*3*num_channels2) ))
    W6 = tf.Variable(tf.truncated_normal([3,3,num_channels3,num_channels3], stddev=1.0/np.sqrt(3*3*num_channels3) ))
    W7 = tf.Variable(tf.truncated_normal([3,3,num_channels3,num_channels3], stddev=1.0/np.sqrt(3*3*num_channels3) ))
    W8 = tf.Variable(tf.truncated_normal([3,3,num_channels3,num_channels3], stddev=1.0/np.sqrt(3*3*num_channels3) ))


    
    W_fc1 = tf.Variable( tf.truncated_normal([3*3*num_channels3, num_H1], stddev= 1.0/np.sqrt(3*3*num_channels3) ) )
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
    P3_flat =  tf.reshape(P3_drop, [-1, 3*3*num_channels3])
    H1 = tf.nn.relu( tf.matmul(P3_flat, W_fc1))
    H1_drop = tf.nn.dropout(H1, keep_prob)
    H2 = tf.nn.relu( tf.matmul(H1_drop, W_fc2))
    H2_drop = tf.nn.dropout(H2, keep_prob)
    u = tf.matmul(H2_drop, W_fc3)

        # W1 = tf.get_variable('D_W1', [5, 5, 3, K], initializer=tf.contrib.layers.xavier_initializer())
        # B1 = tf.get_variable('D_B1', [K], initializer=tf.constant_initializer())

        # W2 = tf.get_variable('D_W2', [5, 5, K, M], initializer=tf.contrib.layers.xavier_initializer())
        # B2 = tf.get_variable('D_B2', [M], initializer=tf.constant_initializer())

        # W3 = tf.get_variable('D_W3', [5, 5, M, N], initializer=tf.contrib.layers.xavier_initializer())
        # B3 = tf.get_variable('D_B3', [N], initializer=tf.constant_initializer())

        # W4 = tf.get_variable('D_W4', [N, 10], initializer=tf.contrib.layers.xavier_initializer())
        # B4 = tf.get_variable('D_B4', [10], initializer=tf.constant_initializer())
        

        # conv1 = conv(X, W1, B1, stride=2, name='conv1')
        # bn1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv1))

        # conv2 = conv(bn1, W2, B2, stride=2, name='conv2')
        # bn2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2))

        # conv3 = conv(bn2, W3, B3, stride=2, name='conv3')
        # bn3 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(conv3)), keep_prob)

        # pooled = tf.nn.avg_pool(bn3,ksize=[1,3,3,1],strides=[1,1,1,1],padding='VALID')
    
        # flat = tf.reshape(pooled,[batch_size, N])
        # output = tf.matmul(flat, W4) + B4

        # # return tf.nn.softmax(output)

        #return output
    return u
