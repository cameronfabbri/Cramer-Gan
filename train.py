import scipy.misc as misc
import time
import tensorflow as tf
from architecture import netD, netG
#from resnet import *
import numpy as np
import random
import ntpath
import sys
import cv2
import os
from skimage import color
import argparse
import data_ops

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--DATASET',    required=True,help='The DATASET to use')
   parser.add_argument('--DATA_DIR',   required=True,help='Directory where data is')
   parser.add_argument('--BATCH_SIZE', required=True,help='Batch size',type=int)
   a = parser.parse_args()

   DATASET        = a.DATASET
   DATA_DIR       = a.DATA_DIR
   BATCH_SIZE     = a.BATCH_SIZE
   CHECKPOINT_DIR = 'checkpoints/'+DATASET+'/'
   IMAGES_DIR     = CHECKPOINT_DIR+'images/'

   try: os.mkdir('checkpoints/')
   except: pass
   try: os.mkdir(CHECKPOINT_DIR)
   except: pass
   try: os.mkdir(IMAGES_DIR)
   except: pass
   
   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   z1          = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z1')
   z2          = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z2')

   train_images_list = data_ops.loadData(DATA_DIR, DATASET)
   filename_queue    = tf.train.string_input_producer(train_images_list)

   # sample from true data
   real_images = data_ops.read_input_queue(filename_queue, BATCH_SIZE)

   # dummy to initialize D
   dummy = netD(real_images, reuse=False)

   # sample two independent images from the generator
   gen_images1 = netG(z1, BATCH_SIZE)
   gen_images2 = netG(z2, BATCH_SIZE, reuse=True)

   # define the critic
   def critic(x):
      return tf.norm(netD(x,reuse=True)-netD(gen_images2,reuse=True), ord=2) - tf.norm(netD(x,reuse=True), ord=2)

   # sample epsilon from uniform distribution
   epsilon = tf.random_uniform([], 0.0, 1.0)
   
   # interpolate real and generated first samples
   x_hat = epsilon * real_images + (1 - epsilon) * gen_images1
   d_hat = critic(x_hat)
   ddx = tf.gradients(d_hat, x_hat)[0]
   ddx = tf.norm(ddx, axis=1)
   ddx = tf.reduce_mean(tf.square(ddx - 1.0) * 10)
   
   D_real = netD(real_images)
   D_gen1 = netD(gen_images1)
   D_gen2 = netD(gen_images2)

   # compute generator loss
   #errG = tf.norm((D_real-D_gen1),ord=2) + tf.norm((D_real-D_gen2),ord=2) - tf.norm((D_gen1-D_gen2),ord=2)

   # computer the surrogate generator loss
   errG = tf.reduce_mean(critic(real_images) - critic(gen_images1))

   # compute the critic loss
   errD = -errG + ddx

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # optimize G
   G_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.,beta2=0.9).minimize(errG, var_list=g_vars, global_step=global_step)

   # optimize D
   D_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.,beta2=0.9).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)

   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/'+'logs/', graph=tf.get_default_graph())

   tf.add_to_collection('G_train_op', G_train_op)
   tf.add_to_collection('D_train_op', D_train_op)
   
   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   ########################################### training portion

   step = sess.run(global_step)
   
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)

   n_critic = 5

   while True:
      
      start = time.time()

      # train the discriminator for 5 or 25 runs
      for critic_itr in range(n_critic):
         batch_z1 = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_z2 = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         sess.run(D_train_op, feed_dict={z1:batch_z1, z2:batch_z2})

      # now train the generator once! use normal distribution, not uniform!!
      batch_z1 = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      batch_z2 = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      sess.run(G_train_op, feed_dict={z1:batch_z1, z2:batch_z2})

      # now get all losses and summary *without* performing a training step - for tensorboard
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={z1:batch_z1, z2:batch_z2})
      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-start
      step += 1
    
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
         batch_z  = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         gen_imgs = sess.run([gen_images], feed_dict={z1:batch_z1, z2:batch_z2})

         data_ops.saveImage(gen_imgs[0], step, IMAGES_DIR)
         print 'Done saving'



