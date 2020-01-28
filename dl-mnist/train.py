

import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from module import generator_multiunet

from datapro_lightmerge_7x7 import get_data_loaders
import tensorboardX
# from tensorboardX.x2num import make_grid

################################################
# For settings
import config
args = config.get_args()

if True:
    args.catagory = 'mnist'
    args.shininess = 0
    args.mergelow = 0.0
    args.mergehigh = 0.2

# always use unet
modelname = 'model'
modelname = '%s-unet' % modelname

#########################################################
in_dim = args.in_dim
out_dim = args.out_dim
gf_dim = args.gf_dim
modelname = '%s-dim%d' % (modelname, gf_dim)

tfim = tf.placeholder(shape=(None, 256, 256, in_dim * 24), dtype=tf.float32)
tfgt = tf.placeholder(shape=(None, 256, 256, out_dim), dtype=tf.float32)

#############################################
# ismultiloss = True
# issquare = True
generator = generator_multiunet
tfre, tfre2, tfre3 = generator(tfim, gf_dim=gf_dim, reuse=False, name='net', output_c_dim=out_dim)

loss = tf.reduce_mean(tf.square(tfgt - tfre))
loss += 0.8 * tf.reduce_mean(tf.square(tf.image.resize_images(tfgt, size=[128, 128]) - tfre2))
loss += 0.6 * tf.reduce_mean(tf.square(tf.image.resize_images(tfgt, size=[64, 64]) - tfre3))

modelname = '%s-mullosssquare' % modelname

##############################################
if __name__ == '__main__':
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    vars = tf.trainable_variables()
    for var in vars:
        print(var.name)
    
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.999)
    varlists = tf.trainable_variables()
    for var in varlists:
        print(var.name)
    
    dldp = optimizer.compute_gradients(loss, var_list=varlists)
    train_op = optimizer.apply_gradients(dldp)
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    ###################################
    catagoryname = args.catagory
    rootfolder = args.datafolder
    
    shi = args.shininess
    if shi > 0:
        ismergewithdiffuse = True
    else:
        ismergewithdiffuse = False
    
    mergelow = args.mergelow
    mergehigh = args.mergehigh
    
    train_loader = get_data_loaders(
        rootfolder, shininess=shi, isrgb=False, \
        ismergewithdiffuse=ismergewithdiffuse, mergelow=mergelow, mergehigh=mergehigh, dropratio=0.2, mode='train', \
        bs=6, numworkers=8)
    
    trainval_loader = get_data_loaders(
        rootfolder, shininess=shi, isrgb=False, \
        ismergewithdiffuse=ismergewithdiffuse, mergelow=mergelow, mergehigh=mergehigh,  dropratio=0.2, mode='train_val', \
        bs=6, numworkers=8)
    
    redir = args.svfolder
    if not os.path.exists(redir):
        os.mkdir(redir)
    
    modeldir = '%s/%s-%s-shiny%d-low%.1f-high%.1f' % (redir, modelname, catagoryname, shi, mergelow, mergehigh)
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    
    ##############################################
    saver = tf.train.Saver(max_to_keep=3)
    
    train_log = tensorboardX.SummaryWriter('%s/train' % modeldir)
    trainval_log = tensorboardX.SummaryWriter('%s/trainval' % modeldir)
    
    epoch = 15
    global_step = 0
    for epo in range(epoch):
        
        # validate
        lossavgnp = 0
        for i, data in enumerate(trainval_loader):
            da = data['imlight']
            feed_dict = {tfgt:data['gt'], tfim:da}
            lossnp, imprenp = sess.run([loss, tfre], feed_dict=feed_dict)
            
            lossavgnp += lossnp
            print('{} {} {}'.format(epo, i, lossnp))
            
            trainval_log.add_scalar('loss', scalar_value=lossnp, global_step=global_step + i)
            if True:
                imtwonp = np.concatenate((imprenp, data['gt']), axis=2)
                imtwonp = np.tile(imtwonp, [1, 1, 1, 3])
                # imtwonp = make_grid(np.transpose(imtwonp, [0, 3, 1, 2]), ncols=2)
                imtwonp = np.concatenate([d for d in imtwonp], axis=0)
                # imtwonp = make_grid(np.transpose(imtwonp, [0, 3, 1, 2]), ncols=2)
                imtwonp = (imtwonp + 1) / 2
                cv2.imwrite('%s/%d-train.png'%(modeldir, epo), imtwonp * 255)
                # trainval_log.add_image('pregt', img_tensor=imtwonp, global_step=global_step + i)
            
        lossavgnp /= i + 1
        
        # train
        lr = 5e-5 / (epo / 3 + 1)
        for i, data in enumerate(train_loader):
            global_step += 1
            
            da = data['imlight']
            feed_dict = {tfgt:data['gt'], tfim:da, learning_rate:lr}
            _, lossnp, imprenp = sess.run([train_op, loss, tfre], feed_dict=feed_dict)
            print('{} {} {} {} {} {} {} {}'.format(epo, i, 'lr', lr, 'trainloss', lossnp, 'last trainvalloss', lossavgnp))
            
            train_log.add_scalar('loss', scalar_value=lossnp, global_step=global_step)
            train_log.add_scalar('lr', scalar_value=lr, global_step=global_step)
            if global_step % 50 == 0:
                imtwonp = np.concatenate((imprenp, data['gt']), axis=2)
                imtwonp = np.tile(imtwonp, [1, 1, 1, 3])
                imtwonp = np.concatenate([d for d in imtwonp], axis=0)
                # imtwonp = make_grid(np.transpose(imtwonp, [0, 3, 1, 2]), ncols=2)
                imtwonp = (imtwonp + 1) / 2
                # train_log.add_image('pregt', img_tensor=imtwonp, global_step=global_step)
                cv2.imwrite('%s/%d-test.png'%(modeldir, epo), imtwonp * 255)
                
        # save
        saver.save(sess, save_path='%s/model' % modeldir, global_step=global_step)
    
    train_log.export_scalars_to_json("%s/train_scalars.json" % modeldir) 
    train_log.close()
    trainval_log.export_scalars_to_json("%s/trainval_scalars.json" % modeldir) 
    trainval_log.close()

