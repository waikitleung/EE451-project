import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference_origin
import mnist_multi_gpu_train

def evaluate(mnist):
    with tf.Graph().as_default():
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
        images = mnist.test.images
        labels = mnist.test.labels
        logits = mnist_inference_origin(images,False,None)
        top_k_op = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_multi_gpu_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            predictions = np.sum(sess.run([top_k_op]))
            print('%s: precision = %.3f' % (datetime.now(), predictions))

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot= True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()
