# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:55:59 2018

@author: 锦
"""
import os
import datetime

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference_origin
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 200
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "logs_and_models/"
MODEL_NAME = "model.ckpt"



def train(mnist):
    x = tf.placeholder(
        tf.float32, [BATCH_SIZE,
                     mnist_inference_origin.IMAGE_SIZE,
                     mnist_inference_origin.IMAGE_SIZE,
                     mnist_inference_origin.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(
        tf.float32, [None,mnist_inference_origin.OUTPUT_NODE], name='y-input')    #attention
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference_origin.inference(x,True,regularizer)
    global_step = tf.Variable(0, trainable=False)
    
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY
            )
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                   .minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    saver = tf.train.Saver()
    
    with tf.Session(config = tf.ConfigProto(log_device_placement=True)) as sess:
    #with tf.Session(config = tf.ConfigProto(device_count={"CPU":8},inter_op_parallelism_threads=1,intra_op_parallelism_threads=1)) as sess:
        tf.initialize_all_variables().run()
        start = datetime.datetime.now()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,(BATCH_SIZE,
                                         mnist_inference_origin.IMAGE_SIZE,
                                         mnist_inference_origin.IMAGE_SIZE,
                                         mnist_inference_origin.NUM_CHANNELS))
            _, loss_value, steps = sess.run([train_op, loss, global_step],
                                            feed_dict={x:reshaped_xs,y_:ys})
            if i % 10 == 0:
                print("After %d training steps, loss on training"
                      "batch is %g." % (steps, loss_value))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME),
                       global_step=global_step)

        end = datetime.datetime.now()
        print (end-start)
    
def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
    train(mnist)
    
if __name__=='__main__':
    tf.app.run()
    
    
    
    
    
    
    
    