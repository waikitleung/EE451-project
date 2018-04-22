# coding=utf-8
from datetime import datetime

import os
import time

import tensorflow as tf
import mnist_inference_origin

# define the variants 
BATCH_SIZE = 100 
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99 
N_GPU = 4

# define program logs and the output route of the model
MODEL_SAVE_PATH = "logs_and_models/"
MODEL_NAME = "model.ckpt"
DATA_PATH = "output.tfrecords" 

# define the input queue 
def get_input():
    filename_queue = tf.train.string_input_producer([DATA_PATH])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

	
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # decode the picture and label message
    decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
    reshaped_image = tf.reshape(decoded_image, [784])
    retyped_image = tf.cast(reshaped_image, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    
    # define the input queue
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    return tf.train.shuffle_batch(
    	[retyped_image, label], 
    	batch_size=BATCH_SIZE, 
    	capacity=capacity, 
    	min_after_dequeue=min_after_dequeue)

# define the loss function
def get_loss(x, y_, regularizer, scope, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        y = mnist_inference_origin.inference(x, True,regularizer)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))
    tf.summary.scalar('cross_entropy',cross_entropy)
    regularization_loss = tf.add_n(tf.get_collection('losses', scope))
    loss = cross_entropy + regularization_loss
    tf.summary.scalar('loss',loss)
    return loss

# calculate each gradient's average
def average_gradients(tower_grads):
    average_grads = []

   
    for grad_and_vars in zip(*tower_grads):
        # calcluate each gradient's average on all GPU
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        # map the variants and their average gradients 
        average_grads.append(grad_and_var)
    # return all the variant's average gradient which are used to update the variants
    return average_grads

# main train process
def main(argv=None): 
    # map the simple calculation on CPU while the complex computation on GPU
    with tf.Graph().as_default(), tf.device('/cpu:0'):
		
		# define the basic train process
        x,y_= get_input()
        x = tf.reshape(x, [-1, 28, 28, 1])
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
        
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(    #指数衰减法衰减学习速率
            LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE, LEARNING_RATE_DECAY)
        tf.summary.scalar('learning_rate', learning_rate)       
        
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        #opt = tf.train.AdagradOptimizer(learning_rate)
        
        tower_grads = []
        reuse_variables = False
        # divide the optimation job on different GPU
        for i in range(N_GPU):
            # map the optimation task on specific GPU
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    cur_loss = get_loss(x, y_, regularizer, scope, reuse_variables)
                    reuse_variables = True
                    grads = opt.compute_gradients(cur_loss)
                    tower_grads.append(grads)
        
        # calculate the variant's average graident
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
            	tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)

        # use average gradients to update the variants
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # calculate the moving average for each variants
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_to_average = (tf.trainable_variables() +tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)
        # for each iteration we need to update the variants and calculate the moving average
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge_all()        
        init = tf.initialize_all_variables()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True)) as sess:
            # initialze all the variants and the input queue
            init.run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)

            start = datetime.now()
            for step in range(TRAINING_STEPS):
                # execute the train program and record the train time
                start_time = time.time()
                _, loss_value = sess.run([train_op, cur_loss])
                duration = time.time() - start_time
                
                # after a period record the progress of the train process 
                if step != 0 and step % 10 == 0:
                    # display the numbers of example to train 
                    num_examples_per_step = BATCH_SIZE * N_GPU
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / N_GPU
    
                    # display the train message
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
                    
                    # display the training progress through TensorBoard
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, step)
    
                # save the model every period
                if step % 1000 == 0 or (step + 1) == TRAINING_STEPS:
                    checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                    saver.save(sess, checkpoint_path, global_step=step)
        
            coord.request_stop()
            coord.join(threads)
            end = datetime.now()
            print (end-start)
        
if __name__ == '__main__':
	tf.app.run()

