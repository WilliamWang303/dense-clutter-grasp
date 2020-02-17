 # -*- coding: utf-8 -*-
import os
import configparser
import argparse
import numpy as np
import signal
import shutil
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import progressbar
import tensorflow as tf

import ae_factory as factory
import utils as u
from vgg import vgg_arg_scope, vgg_16

slim = tf.contrib.slim


def main():

    workspace_path = '/home/william/Documents/Master/AAE_Workspace'
    
    gentle_stop = np.array((1,), dtype=np.bool)
    gentle_stop[0] = False
    def on_ctrl_c(signal, frame):
        gentle_stop[0] = True
    signal.signal(signal.SIGINT, on_ctrl_c)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("-d", action='store_true', default=False)
    arguments = parser.parse_args()
    
    full_name = arguments.experiment_name.split('/')
    
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    
    debug_mode = arguments.d
    

    cfg_file_path = u.get_config_file_path(workspace_path, experiment_name, experiment_group)
    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)
    checkpoint_file = u.get_checkpoint_basefilename(log_dir)
    ckpt_dir = u.get_checkpoint_dir(log_dir)
    train_fig_dir = u.get_train_fig_dir(log_dir)
    dataset_path = '/home/william/Documents/Master/LNC/pose_dataset/Views'
    
    if not os.path.exists(cfg_file_path):
        print 'Could not find config file:\n'
        print '{}\n'.format(cfg_file_path)
        exit(-1)
        
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(train_fig_dir):
        os.makedirs(train_fig_dir)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    shutil.copy2(cfg_file_path, log_dir)

    # build graph
    with tf.variable_scope(experiment_name):
        dataset = factory.build_dataset(dataset_path, args)
        queue = factory.build_queue(dataset, args) # put dataset into queue
        encoder = factory.build_encoder(queue.x, queue.y, args, is_training=True)
        decoder = factory.build_decoder(queue.y, encoder, args, is_training=True)
        ae = factory.build_ae(encoder, decoder, args) # encoder and decoder graph
        codebook = factory.build_codebook(encoder, dataset, args)
        # train_op = factory.build_train_op(ae, args) # optimizer of ae.loss
        # saver = tf.train.Saver(save_relative_paths=True)
    
    # build pretrained vgg network
    recons = tf.image.resize_images(decoder.x,[224,224])
    target = tf.image.resize_images(decoder.reconstruction_target,[224,224])
    vgg_x = tf.concat([recons,target], 0)
    
    arg_scope = vgg_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        _, endpoints_dict = vgg_16(vgg_x, 1, is_training=False, spatial_squeeze=False)
    
    content_loss = 0
    content_layers = ["vgg_16/conv1/conv1_2","vgg_16/conv2/conv2_2"]
    for layer in content_layers:
        generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images) # num of elements
        content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  # remain the same as in the paper
    
    total_loss = ae.loss + 0.1 * content_loss # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    tf.summary.scalar('total_loss', total_loss)

    variable_to_train = []
    for variable in tf.trainable_variables():
        if not(variable.name.startswith('vgg_16')):
            variable_to_train.append(variable)
    train_op = tf.train.AdamOptimizer(2e-4).minimize(total_loss, global_step=ae.global_step, var_list=variable_to_train)
    print('Training Variable : ',variable_to_train)

    variables_to_restore = []
    for v in tf.global_variables():
        if not(v.name.startswith('vgg_16')):
            variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, save_relative_paths=True)
    print('Training Restore : ',variables_to_restore)

    exclusions = ['vgg_16/fc']
    vgg_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            vgg_restore.append(var)
    print('Vgg Restore : ',vgg_restore)
    restore_vgg = slim.assign_from_checkpoint_fn("pretrained/vgg_16.ckpt",vgg_restore,ignore_missing_vars=True)


    num_iter = args.getint('Training', 'NUM_ITER') if not debug_mode else np.iinfo(np.int32).max
    save_interval = args.getint('Training', 'SAVE_INTERVAL')
    val_interval = args.getint('Training', 'VAL_INTERVAL')
    model_type = args.get('Dataset', 'MODEL')
    
    widgets = ['Training: ', progressbar.Percentage(),
         ' ', progressbar.Bar(),
         ' ', progressbar.Counter(), ' / %s' % num_iter,
         ' ', progressbar.ETA(), ' ']
    bar = progressbar.ProgressBar(maxval=num_iter,widgets=widgets)


    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.9)
    config = tf.ConfigProto(gpu_options=gpu_options)
    batch_size = args.getint('Training', 'BATCH_SIZE')

    # prepare for validation set
    '''
    validation_dir = '../test/data_grasp'
    val_img_list = os.listdir(validation_dir)
    x_val = []
    for val_img in val_img_list:
        val_im = cv2.imread(os.path.join(validation_dir,val_img))
        x_val.append(val_im/255.)
    x_val = np.array(x_val,np.float)
    '''

    # start training
    with tf.Session(config=config) as sess:
        
        # restoring vgg
        restore_vgg(sess)
        print('already initilizing vgg')
        
        # restoring model or initilizing model
        chkpt = tf.train.get_checkpoint_state(ckpt_dir)
        if chkpt and chkpt.model_checkpoint_path:
            print chkpt.model_checkpoint_path
            saver.restore(sess, chkpt.model_checkpoint_path)
        else:
            sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        merged_loss_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(ckpt_dir, sess.graph)
                
        if not debug_mode:
            print 'Training with %s model' % args.get('Dataset','MODEL'), os.path.basename(args.get('Paths','MODEL_PATH'))
            bar.start()
        # ------------------------------- start training --------------------
        queue.start(sess) # putting data into queue
        for i in xrange(ae.global_step.eval(), num_iter):
            if not debug_mode:
                sess.run(train_op)
                if i % 10 == 0:
                    loss = sess.run(merged_loss_summary)
                    summary_writer.add_summary(loss, i)
                '''
                if i % val_interval == 0:
                    print "starting validation"
                    codebook.update_embedding_grasp(sess,batch_size)
                    idcs = codebook.nearest_rotation_batch(sess, x_val)
                    reconst = sess.run(decoder.x,feed_dict={queue.x:x_val})
                    # print idcs
                    val_loss_list = []
                    for index in range(len(idcs)):
                        pose_img = cv2.imread(os.path.join(dataset_path,'{}.png'.format(idcs[index])))
                        pose_img = cv2.bilateralFilter(pose_img,7,100,100)
                        pose_img = pose_img.astype(np.float)/255.
                        val_loss = np.mean(np.square(reconst[index] - pose_img))
                        val_loss_list.append(val_loss)
                    val_loss_ = np.mean(np.array(val_loss_list))
                    print 'val loss = {}'.format(val_loss_)
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag="val_loss", simple_value=val_loss_)
                    ])
                    summary_writer.add_summary(summary, i)
                '''

                bar.update(i)
                if (i+1) % save_interval == 0:
                    # save model and reconstruct result
                    codebook.update_embedding_grasp(sess,batch_size)
                    saver.save(sess, checkpoint_file, global_step=ae.global_step)

                    this_x, this_y = sess.run([queue.x, queue.y])
                    reconstr_train = sess.run(decoder.x,feed_dict={queue.x:this_x})
                    train_imgs = np.hstack(( u.tiles(this_x, 4, 4), u.tiles(reconstr_train, 4,4),u.tiles(this_y, 4, 4)))
                    cv2.imwrite(os.path.join(train_fig_dir,'training_images_%s.png' % i), train_imgs*255)
            else:
                

                this_x, this_y = sess.run([queue.x, queue.y])
                reconstr_train = sess.run(decoder.x,feed_dict={queue.x:this_x})
                
                cv2.imshow('sample batch', np.hstack(( u.tiles(this_x, 3, 3), u.tiles(reconstr_train, 3,3),u.tiles(this_y, 3, 3))) )
                k = cv2.waitKey(0)
                if k == 27:
                    break

            if gentle_stop[0]:
                break

        queue.stop(sess)
        if not debug_mode:
            bar.finish()
        if not gentle_stop[0] and not debug_mode:
            print 'To create the embedding run:\n'
            print 'ae_embed {}\n'.format(full_name)

if __name__ == '__main__':
    
    main()
    
