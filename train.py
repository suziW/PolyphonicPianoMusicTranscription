#00!/home/suzi/anaconda3/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf 
import load

window_size = 440
dirpath_x = 'model/data/x_input.dat'
dirpath_y = 'model/data/y_input.dat'
lr = 0.001
epochs = 1000
batch_size = 128
dropout = 0.75
num_steps = 4400*10 

data = load.DataGen(dirpath_x, dirpath_y, batch_size=batch_size)
x_train_shape, y_train_shape, positive_train = data.getinfo_train()
x_test_shape, y_test_shape, positive_test = data.getinfo_test()
print(data.getinfo_train())
print(data.getinfo_test())

X = tf.placeholder(tf.float32, [None, x_train_shape[0]])
Y = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

def conv1d(x, w, b, stride=1):
    x = tf.nn.conv1d(x, w, stride, 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, window_size, 1])
    # x_window = tf.multiply(x, weights['ww'])
    conv1 = conv1d(x, weights['wc1'], biases['bc1'])
    conv2 = conv1d(conv1, weights['wc2'], biases['bc1'])

    fc1 = tf.reshape(conv2, [-1, weights['wd1'][0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    
    return out

weights = {
    'wc1': tf.Variable(tf.random_normal([110, 1, 88])),
    'wc2': tf.Variable(tf.random_normal([110, 88, 352])),
    'wd1': tf.Variable(tf.random_normal([window_size*352, 352])),
    'wd2': tf.Variable(tf.random_normal([352, 88])),
    'out': tf.Variable(tf.random_normal([88, 1]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([88])),
    'bc2': tf.Variable(tf.random_normal([352])),
    'bd1': tf.Variable(tf.random_normal([352])),
    'bd2': tf.Variable(tf.random_normal([88])),
    'out': tf.Variable(tf.random_normal([1])) 
}

logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.sigmoid(logits) 
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train = optimizer.minimize(loss)

correct_pred = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = data.train_gen().next()
        # Run optimization op (backprop)
        sess.run(train, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        if step % 100 == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: data.get_test_data()[0],
                                      Y: data.get_test_data()[1],
                                      keep_prob: 1.0}))
