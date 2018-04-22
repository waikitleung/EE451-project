from datetime import datetime
import os
import time

import tensorflow as tf
import mnist_inference


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.9
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99
N_GPU = 4
MODEL_SAVE_PATH = "/home/ubuntu/"
MODEL_NAME = "model.ckpt"


DATA_PATH = "/home/ubuntu/output.tfrecords"

def get_input():
	filename_queue = tf.train.string_input_producer([DATA_PATH])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(
		serialized_example,
		features={
			'image_raw':tf.FixedLenFeature([],tf.string),
			'pixels':tf.FixedLenFeature([],tf.int64),
			'label':tf.FixedLenFeature([],tf.int64),
		})

	decoded_image = tf.decode_raw(features['image_raw'],tf.uint8)
	reshaped_image = tf.reshape(decoded_image,[784])
	retyped_image = tf.cast(reshaped_image,tf.float32)
	label = tf.cast(features['label'], tf.int32)

	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3*BATCH_SIZE
	return tf.train.shuffle_batch(
		[retyped_image,label],
		batch_size = BATCH_SIZE,
		capacity = capacity,
		min_after_dequeue = min_after_dequeue)
    
    
def get_loss(x, y_, regularizer,scope):							#258_begin
	y = mnist_inference.inference(x,regularizer)
	cross_entropy = tf.reduce_mean(
		tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_))
	regularization_loss = tf.add_n(tf.get_collection('losses',scope))
	loss = cross_entropy + regularization_loss
	return loss
	
def average_gradients(tower_grads):
		average_grads = []
		for grad_and_vars in zip(*tower_grads):
			grads = []
			for g, _ in grad_and_vars:
				expanded_g =tf.expand_dims(g, 0)
				grads.append(expanded_g)
			grad = tf.concat(grads,0)    #Attention the order is important
			grad = tf.reduce_mean(grad, 0)
			
			v = grad_and_vars[0][1]
			grad_and_var = (grad, v)
			average_grads.append(grad_and_var)
		return average_grads
	

def main(argv=None):
	with tf.Graph().as_default(),tf.device('/cpu:0'):
		x, y_ = get_input()
		regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
		
		global_step = tf.get_variable(
			'global_step', [], initializer=tf.constant_initializer(0),
			trainable=False)
		learning_rate = tf.train.exponential_decay(
			LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE,		#259_begin
			LEARNING_RATE_DECAY)
			
		opt = tf.train.GradientDescentOptimizer(learning_rate)
		
		tower_grads =[]
		for i in range(N_GPU):
			with tf.device('/gpu:%d' % i):
				with tf.name_scope('GPU_%d' % i) as scope:
					cur_loss = get_loss(x, y_, regularizer, scope)
					tf.get_variable_scope().reuse_variables()
				
					grads = opt.compute_gradients(cur_loss)
					tower_grads.append(grads)
					
		grads = average_gradients(tower_grads)
		for grad, var in grads:
			if grad is not None:
			 tf.summary.histogram(
				'gradients_on_average/%s' % var.op.name, grad)
				
		apply_gradient_op = opt.apply_gradients(
			grads, global_step=global_step)
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)
			
		variable_averages = tf.train.ExponentialMovingAverage(
			MOVING_AVERAGE_DECAY, global_step)
		variable_averages_op =variable_averages.apply(
			tf.trainable_variables())
			
		train_op = tf.group(apply_gradient_op, variables_averages_op)
		saver = tf.train.Saver(tf.all_variables())
		summary_op = tf.merge_all_summaries()
		init = tf.initialize_all_variables()
		with tf.Session(config = tf.ConfigProto(
			allow_soft_placement=True, 
			log_device_placement=True)) as sess:
			init.run()
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess,coord=coord)
			summary_writer = tf.train.SummaryWriter(MODEL_SAVE_PATH,sess.graph)

			for step in range(TRAINING_STEPS):
				start_time = time.time()
				_,loss_value = sess.run([train_op,cur_loss])
				duration = time.time()-start_time

				if step !=0 and step%10 == 0:
					num_examples_per_step = BATCH_SIZE*N_GPU
					examples_per_sec = num_examples_per_step / duration
					sec_per_batch = duration / N_GPU

					format_str = ('step %d, loss =%.2f(%.1f examples/''sec;%.3f sec/batch)')
					print(format_str %(step,loss_value,examples_per_sec,sec_per_batch))
					summary = sess.run(summary_op)
					summary_writer.add_summary(summary,step)
				if step%1000 == 0 or (step+1)==TRAINING_STEPS:
					checkpoint_path = os.path.join(MODEL_SAVE_PATH,MODEL_NAME)
					saver.save(sess,checkpoint_path,global_step=step)
			coord.request_stop()
			coord.join(threads)
if __name__ == '__main__':
	tf.app.run()