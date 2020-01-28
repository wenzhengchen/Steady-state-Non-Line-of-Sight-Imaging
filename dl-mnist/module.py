

import numpy as np
import tensorflow as tf
from ops import instance_norm, conv2d, deconv2d, lrelu


######################################################################
def generator_multiunet(image, gf_dim, reuse=False, name="generator", output_c_dim=-1, istraining=True):

    if istraining:
        dropout_rate = 0.5
    else:
        dropout_rate = 1.0
    
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv2d(image, gf_dim, name='g_e1_conv'), 'g_bn_e1')
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), gf_dim * 2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), gf_dim * 4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv2d(lrelu(e3), gf_dim * 8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), gf_dim * 8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), gf_dim * 8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        
        e7 = instance_norm(conv2d(lrelu(e6), gf_dim * 8, ks=3, s=1, padding='VALID', name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        
        e8 = instance_norm(conv2d(lrelu(e7), gf_dim * 16, ks=2, s=1, padding='VALID', name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), gf_dim * 8, ks=2, s=1, padding='VALID', name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), gf_dim * 8, ks=3, s=1, padding='VALID', name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), gf_dim * 8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), gf_dim * 8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), gf_dim * 4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)
        
        d6 = deconv2d(tf.nn.relu(d5), gf_dim * 2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)
        
        d6_pre = deconv2d(tf.nn.relu(d5), output_c_dim, name='g_d6_pre')
        # d6_pre is (64 x 64 x output_c_dim)
        
        d7_pre = deconv2d(tf.nn.relu(d6), output_c_dim, name='g_d7_pre')
        # d7_pre is (128 x 128 x output_c_dim)
        
        d8_pre = deconv2d(tf.nn.relu(d7), output_c_dim, name='g_d8_pre')
        # d8_pre is (256 x 256 x output_c_dim)
        
        return tf.nn.tanh(d8_pre), tf.nn.tanh(d7_pre), tf.nn.tanh(d6_pre)

