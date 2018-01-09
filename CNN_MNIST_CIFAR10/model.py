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
        K = 64
        M = 128
        N = 256

        W1 = tf.get_variable('D_W1', [5, 5, 3, K], initializer=tf.contrib.layers.xavier_initializer())
        B1 = tf.get_variable('D_B1', [K], initializer=tf.constant_initializer())

        W2 = tf.get_variable('D_W2', [5, 5, K, M], initializer=tf.contrib.layers.xavier_initializer())
        B2 = tf.get_variable('D_B2', [M], initializer=tf.constant_initializer())

        W3 = tf.get_variable('D_W3', [5, 5, M, N], initializer=tf.contrib.layers.xavier_initializer())
        B3 = tf.get_variable('D_B3', [N], initializer=tf.constant_initializer())

        W4 = tf.get_variable('D_W4', [N, 10], initializer=tf.contrib.layers.xavier_initializer())
        B4 = tf.get_variable('D_B4', [10], initializer=tf.constant_initializer())
        

        conv1 = conv(X, W1, B1, stride=2, name='conv1')
        bn1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv1))

        conv2 = conv(bn1, W2, B2, stride=2, name='conv2')
        bn2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2))

        conv3 = conv(bn2, W3, B3, stride=2, name='conv3')
        bn3 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(conv3)), keep_prob)

        pooled = tf.nn.avg_pool(bn3,ksize=[1,3,3,1],strides=[1,1,1,1],padding='VALID')
    
        flat = tf.reshape(pooled,[batch_size, N])
        output = tf.matmul(flat, W4) + B4

        # return tf.nn.softmax(output)
        return output