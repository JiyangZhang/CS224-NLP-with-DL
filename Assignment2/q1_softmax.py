import tensorflow as tf
import numpy as np

def softmax(x):
	x_max = tf.reduce_max(x,1,keep_dims=True)
	x_sub = tf.subtract(x,x_max)
	nominator = tf.exp(x)
	denominator = tf.reduce_sum(nominator, 1, keep_dims=True)
	result = nominator / denominator
	return result


def cross_entropy_loss(y, y_hat):
	"""
	suppose the y and y_hat is [BS, n_class]
	"""
	y_log = tf.log(y_hat)
	cross_entropy = -1 * tf.reduce_sum(tf.to_float(y) * y_log, 1, keep_dims=True)
	return cross_entropy



if __name__ == '__main__':
	sess = tf.Session()
	test_softmax()
	