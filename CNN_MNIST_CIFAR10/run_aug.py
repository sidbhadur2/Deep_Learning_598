from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange 
import tensorflow as tf
import numpy as np
import time

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

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
        num_channels3 = 64*4
        num_H1 = 20
        num_H2 = 200

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


            
        W_fc1 = tf.Variable( tf.truncated_normal([3*3*num_channels3, num_H1], stddev= 1.0/np.sqrt(3*3*num_channels3) ) )
            #     W_fc2 = tf.Variable( tf.truncated_normal([num_H2,10], stddev= 1.0/np.sqrt(num_H1) ) )   
        W_fc2 = tf.Variable( tf.truncated_normal([num_H1,num_H2], stddev= 1.0/np.sqrt(num_H1) ) )
        W_fc3 = tf.Variable( tf.truncated_normal([num_H2,10], stddev= 1.0/np.sqrt(num_H2) ) )


        #NETWORK
        C1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = "SAME") )
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
        #W1 = tf.get_variable('D_W1', [5, 5, 3, K], initializer=tf.contrib.layers.xavier_initializer())
        #B1 = tf.get_variable('D_B1', [K], initializer=tf.constant_initializer())

        #W2 = tf.get_variable('D_W2', [5, 5, K, M], initializer=tf.contrib.layers.xavier_initializer())
        #B2 = tf.get_variable('D_B2', [M], initializer=tf.constant_initializer())

        #W3 = tf.get_variable('D_W3', [5, 5, M, N], initializer=tf.contrib.layers.xavier_initializer())
        #B3 = tf.get_variable('D_B3', [N], initializer=tf.constant_initializer())

        #W4 = tf.get_variable('D_W4', [N, 10], initializer=tf.contrib.layers.xavier_initializer())
        #B4 = tf.get_variable('D_B4', [10], initializer=tf.constant_initializer())
        

        #conv1 = conv(X, W1, B1, stride=2, name='conv1')
        #bn1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv1))

        #conv2 = conv(bn1, W2, B2, stride=2, name='conv2')
        #bn2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2))

        #conv3 = conv(bn2, W3, B3, stride=2, name='conv3')
        #bn3 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(conv3)), keep_prob)

        #pooled = tf.nn.avg_pool(bn3,ksize=[1,3,3,1],strides=[1,1,1,1],padding='VALID')
    
        #flat = tf.reshape(pooled,[batch_size, N])
        #output = tf.matmul(flat, W4) + B4

        # return tf.nn.softmax(output)
        #return output
        return u

def read_cifar10(filename_queue):
    """Reads and parses examples from CIFAR10 data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
        filename_queue: A queue of strings with the filenames to read from.

    Returns:
        An object representing a single example, with the following fields:
            height: number of rows in the result (32)
            width: number of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number
                for this example.
            label: an int32 Tensor with the label in the range 0..9.
            uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
            tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
            tf.strided_slice(record_bytes, [label_bytes],
                                             [label_bytes + image_bytes]),
            [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 8
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)

    return images, tf.reshape(label_batch, [batch_size])

def get_inputs(data_dir, batch_size, is_test=False):
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    if not is_test:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                                 for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    if not is_test:
        ##### NO AUGMENTATION
        # distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)

        ##### AUGMENTATION
        # randomly crop 24x24x3 out of the 32x32x3 input
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
        # flip the image
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # random brightness
        distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
        # random contrast
        distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
    else:
        # crops just the center
        distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)

    # normalizes the inputs (originally 0-255)
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    if not is_test:
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                                         min_fraction_of_examples_in_queue)
        print ('Filling training queue with %d CIFAR images before starting to train. '
                     'This will take a few minutes.' % min_queue_examples)
    else:
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                                                         min_fraction_of_examples_in_queue)
        print ('Filling testing queue with %d CIFAR images.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image,read_input.label,min_queue_examples,batch_size,shuffle=True)

def train_inputs(batch_size):
    data_dir = 'cifar10_data/cifar-10-batches-bin/'
    images, labels = get_inputs(data_dir=data_dir,batch_size=batch_size)
    return images, labels

def test_inputs(batch_size):
    data_dir = 'cifar10_data/cifar-10-batches-bin/'
    images, labels = get_inputs(data_dir=data_dir,batch_size=batch_size, is_test=True)
    return images, labels

import numpy as np
import os

#from model import cifar10_conv

batch_size = 64

#### TRAIN
with tf.device('/cpu:0'):
    images, labels = train_inputs(batch_size)
    images_test, labels_test = test_inputs(batch_size)

with tf.variable_scope('placeholder'):
    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(name='label',dtype=tf.float32,shape=[batch_size,10])
    keep_prob = tf.placeholder(tf.float32 ,shape=())

with tf.variable_scope('model'):
    output = cifar10_conv(X, keep_prob=keep_prob)

with tf.variable_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y))

with tf.variable_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tvar = tf.trainable_variables()
cifar10_var = [var for var in tvar if 'cifar10_conv' in var.name]

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss, var_list=cifar10_var)

saver = tf.train.Saver(tvar)

sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
sess.run(init)


if not os.path.exists('current_model/'):
    os.makedirs('current_model/')

# saver.restore(sess,tf.train.latest_checkpoint('current_model/'))

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

keep_probability = 0.55

tf.train.start_queue_runners()
loss_print = 0
accuracy_print = 0
t = time.time()
for i in range(0,30000000):

    X_batch, labels_batch = sess.run([images, labels])

    y_batch = np.zeros((batch_size,NUM_CLASSES))
    y_batch[np.arange(batch_size),labels_batch] = 1

    _, loss_print, accuracy_print = sess.run([train_step, loss, accuracy], feed_dict={X: X_batch, y: y_batch, keep_prob:keep_probability})

    if i % 20 == 0:
        print('time: %f iteration:%d loss:%f accuracy:%f' % (float(time.time()-t), i, loss_print, accuracy_print))
        t = time.time()

    if i % 500 == 0:

        test_accuracy = 0.0
        accuracy_count = 0

        for j in xrange(50):
            X_batch, labels_batch = sess.run([images_test,labels_test])
            y_batch = np.zeros((batch_size,NUM_CLASSES))
            y_batch[np.arange(batch_size),labels_batch] = 1

            accuracy_print = sess.run([accuracy], feed_dict={X: X_batch, y: y_batch, keep_prob:1.0})

            test_accuracy += accuracy_print[0]
            accuracy_count += 1
        test_accuracy = test_accuracy/accuracy_count
        print('TEST:%f' % test_accuracy)

saver.save(sess, 'current_model/model',global_step=30000)


### JUST TEST
tf.train.start_queue_runners()
t = time.time()
test_accuracy = 0.0
accuracy_count = 0
for i in range(10000):
    X_batch, labels_batch = sess.run([images_test, labels_test])
    
    y_batch = np.zeros((batch_size,NUM_CLASSES))
    y_batch[np.arange(batch_size),labels_batch] = 1

    accuracy_print = sess.run([accuracy], feed_dict={X: X_batch, y: y_batch, keep_prob:1.0})

    test_accuracy += accuracy_print[0]
    accuracy_count += 1
    if i % 10 == 0:
        print('time: %f accuracy:%f' % (float(time.time()-t),test_accuracy/accuracy_count))
        t = time.time()