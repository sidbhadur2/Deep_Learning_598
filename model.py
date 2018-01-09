import tensorflow as tf

def conv(x, w, b, stride, name, padding='SAME'):
    with tf.variable_scope('conv'):
        return tf.nn.conv2d(x,
                           filter=w,
                           strides=[1, stride, stride, 1],
                           padding=padding,
                           name=name) + b


def deconv(x, w, b, shape, stride, name):
    with tf.variable_scope('deconv'):
        return tf.nn.conv2d_transpose(x,
                                       filter=w,
                                       output_shape=shape,
                                       strides=[1, stride, stride, 1],
                                       padding='SAME',
                                       name=name) + b

def lrelu(x, alpha=0.2):
    with tf.variable_scope('leakyReLU'):
        return tf.maximum(x, alpha * x)



def discriminator(X, keep_prob, is_train=True, reuse=False):
    with tf.variable_scope('discriminator'):
        if reuse:
            tf.get_variable_scope().reuse_variables()


        batch_size = tf.shape(X)[0]
        K = 96
        M = 192
        N = 384

        W1 = tf.get_variable('D_W1', [5, 5, 3, K], 
            initializer=tf.contrib.layers.xavier_initializer())
        B1 = tf.get_variable('D_B1', [K], 
            initializer=tf.constant_initializer())

        W2 = tf.get_variable('D_W2', [5, 5, K, M], 
            initializer=tf.contrib.layers.xavier_initializer())
        B2 = tf.get_variable('D_B2', [M], 
            initializer=tf.constant_initializer())

        W3 = tf.get_variable('D_W3', [5, 5, M, N], 
            initializer=tf.contrib.layers.xavier_initializer())
        B3 = tf.get_variable('D_B3', [N], 
            initializer=tf.constant_initializer())

        W4 = tf.get_variable('D_W4', [3, 3, N, N], 
            initializer=tf.contrib.layers.xavier_initializer())
        B4 = tf.get_variable('D_B4', [N], 
            initializer=tf.constant_initializer())

        W5 = tf.get_variable('D_W5', [4, 4, N, N], 
            initializer=tf.contrib.layers.xavier_initializer())
        B5 = tf.get_variable('D_B5', [N], 
            initializer=tf.constant_initializer())

        W6 = tf.get_variable('D_W6', [N, 10+1], 
            initializer=tf.contrib.layers.xavier_initializer())
        

        conv1 = conv(X, W1, B1, stride=2, name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        conv2 = conv(tf.nn.dropout(lrelu(bn1), keep_prob),
            W2, B2, stride=2, name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        conv3 = conv(tf.nn.dropout(lrelu(bn2),keep_prob),
            W3, B3, stride=2, name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        conv4 = conv(tf.nn.dropout(lrelu(bn3),keep_prob),
            W4, B4, stride=1, name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        conv5 = conv(lrelu(bn4), W5, B5, stride=1, name='conv5', padding='VALID')
        bn5 = tf.contrib.layers.batch_norm(conv5,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        flat = tf.reshape(lrelu(bn5),[batch_size,N])
        output = tf.matmul(flat, W6)

        return tf.nn.softmax(output), output, flat


def generator(Z, keep_prob, is_train=True):
    with tf.variable_scope('generator'):

        batch_size = tf.shape(Z)[0]
        K = 512
        L = 256
        M = 128
        N = 64

        W1 = tf.get_variable('G_W1', [100, 4*4*K],
            initializer=tf.contrib.layers.xavier_initializer())
        B1 = tf.get_variable('G_B1', [4*4*K], initializer=tf.constant_initializer())

        W2 = tf.get_variable('G_W2', [4, 4, L, K], 
            initializer=tf.contrib.layers.xavier_initializer())
        B2 = tf.get_variable('G_B2', [L], 
            initializer=tf.constant_initializer())

        W3 = tf.get_variable('G_W3', [6, 6, M, L], 
            initializer=tf.contrib.layers.xavier_initializer())
        B3 = tf.get_variable('G_B3', [M], 
            initializer=tf.constant_initializer())

        W4 = tf.get_variable('G_W4', [6, 6, N, M], 
            initializer=tf.contrib.layers.xavier_initializer())
        B4 = tf.get_variable('G_B4', [N], 
            initializer=tf.constant_initializer())

        W5 = tf.get_variable('G_W5', [3, 3, N, 3], 
            initializer=tf.contrib.layers.xavier_initializer())
        B5 = tf.get_variable('G_B5', [3], 
            initializer=tf.constant_initializer())

        Z = lrelu(tf.matmul(Z, W1) + B1)
        Z = tf.reshape(Z, [batch_size, 4, 4, K])
        Z = tf.contrib.layers.batch_norm(Z,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        deconv1 = deconv(Z, 
            W2, B2, shape=[batch_size, 8, 8, L], stride=2, name='deconv1')
        bn1 = tf.contrib.layers.batch_norm(deconv1,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        deconv2 = deconv(lrelu(bn1), 
            W3, B3, shape=[batch_size, 16, 16, M], stride=2, name='deconv2')
        bn2 = tf.contrib.layers.batch_norm(deconv2,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        deconv3 = deconv(lrelu(bn2), 
            W4, B4, shape=[batch_size, 32, 32, N], stride=2, name='deconv3')
        bn3 = tf.contrib.layers.batch_norm(deconv3,
            is_training=is_train,center=True,decay=0.9,updates_collections=None)

        conv4 = conv(lrelu(bn3), W5, B5, stride=1, name='conv4')
        output = tf.nn.tanh(conv4)

        return output