# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference_origin
import mnist_multi_gpu_train

EVAL_INTERVAL_SECS = 10
BATCH_SIZE = 10000

def evaluate(mnist):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(
        tf.float32, [BATCH_SIZE,
                     mnist_inference_origin.IMAGE_SIZE,
                     mnist_inference_origin.IMAGE_SIZE,
                     mnist_inference_origin.NUM_CHANNELS], name='x-input')
		y_ = tf.placeholder(
			tf.float32, [None, mnist_inference_origin.OUTPUT_NODE], name='y-input')
		xorigin = mnist.test.images
		x0 = tf.reshape(xorigin, [-1, 28, 28, 1])
		y = mnist_inference_origin.inference(x, False, None)
		
		correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		variable_averages = tf.train.ExponentialMovingAverage(
			mnist_multi_gpu_train.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)
		
		
		with tf.Session() as sess:
				ys = mnist.test.labels
				xs = sess.run(x0)
				ckpt = tf.train.get_checkpoint_state(
				mnist_multi_gpu_train.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path\
										  .split('/')[-1].split('-')[-1]
					accuracy_score = sess.run(accuracy,
											feed_dict={x:xs,y_:ys})
					print("After %s train step(s),validation "
							"accuracy = %g" % (global_step, accuracy_score))
				else:
						print('No checkpoint file found')
						
								

				
def main(argv=None):
	mnist = input_data.read_data_sets(".", one_hot= True)
	evaluate(mnist)
		
if __name__ == '__main__':
	tf.app.run()
				
				
				
				
				
				
				
				
				
				