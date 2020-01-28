

import tensorflow as tf


#####################################
epsilon = 1e-8

ini1 = tf.random_normal_initializer(mean=1.0, stddev=0.01, dtype=tf.float32)
ini0 = tf.constant_initializer(value=0.0, dtype=tf.float32)
ini = tf.contrib.layers.xavier_initializer()


#####################################################################
def instance_norm(input_, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input_.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=ini1)
        offset = tf.get_variable("offset", [depth], initializer=ini0)
        mean, variance = tf.nn.moments(input_, axes=[1, 2], keep_dims=True)
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_ - mean) * inv
        return scale * normalized + offset


def conv2d(input_, output_dim, ks=4, s=2, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return tf.layers.conv2d(input_, output_dim, ks, s, padding=padding,
            kernel_initializer=ini)


def deconv2d(input_, output_dim, ks=4, s=2, padding='SAME', name="deconv2d"):
    with tf.variable_scope(name):
        return tf.layers.conv2d_transpose(input_, output_dim, ks, s, padding=padding,
            kernel_initializer=ini)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

