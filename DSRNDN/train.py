#!/usr/bin/env python

#import libs.nets.network as network
import libs.nets.R_DSR_D as network
import libs.datasets.data_factory as dataset
from libs.config.config import *

import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import time
import os


# using GPU numbered 0
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def restore(sess):
     if FLAGS.restore_previous_if_exists:
        try:
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            restorer = tf.train.Saver()
            restorer.restore(sess, checkpoint_path)
            print ('restored previous model %s from %s'\
                    %(checkpoint_path, FLAGS.train_dir))
            time.sleep(2)
            return
        except:
            print ('--restore_previous_if_exists is set, but failed to restore in %s %s'\
                    % (FLAGS.train_dir, checkpoint_path))
            time.sleep(2)

def train():
    print(FLAGS.data_dir) #/data/
    name, seqLen, seq_feature, pair_feature, label = \
        dataset.get_dataset('train', FLAGS.data_dir) # train  /data/

    data_queue = tf.RandomShuffleQueue(capacity=32, min_after_dequeue=16,
            dtypes=(name.dtype, seqLen.dtype,
                seq_feature.dtype, pair_feature.dtype, label.dtype))
    enqueue_op = data_queue.enqueue((name, seqLen, seq_feature, pair_feature, label))
    data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
    (name, seqLen, seq_feature, pair_feature, label) = data_queue.dequeue()
    input_1d = tf.reshape(seq_feature, (1, seqLen,1310))

    input_2d = tf.reshape(pair_feature, (1, seqLen, seqLen, 443))
    label = tf.reshape(label, (1, seqLen, seqLen))
    is_training = tf.placeholder(tf.bool)
    output = network.build(is_training,input_1d, input_2d, label,
            FLAGS.filter_size_1d, FLAGS.filter_size_2d,
            FLAGS.block_num_1d, FLAGS.block_num_2d,
            regulation=True, batch_norm=True)
    print("output:",output)

    prob = output['output_prob']
    loss = output['loss']
    print("prob:",prob)
    print("loss:",loss)

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init_op = tf.group(tf.global_variables_initializer(),
            tf.local_variables_initializer())
    sess.run(init_op)

    # save log
    summary_op = tf.summary.merge_all()
    logdir = os.path.join(FLAGS.train_dir, strftime('%Y%m%d%H%M%S', gmtime()))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

    #restore model
    restore(sess)

    # main loop
    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    tf.train.start_queue_runners(sess=sess, coord=coord)

    feed_dict = {is_training:True}

    saver = tf.train.Saver(max_to_keep=20)
    # train iteration
    for step in range(FLAGS.max_iters):
        _, ids, L, los, output_prob = \
                sess.run([train_step, name, seqLen, loss, prob], feed_dict={is_training:True})
        print ("iter %d: id = %s, seqLen = %3d, loss = %.4f" %(step, ids, L, los))

        if step % 100 == 0:
            summary_str = sess.run(summary_op, feed_dict={is_training:True})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        if (step % 10000 == 0 or step + 1 == FLAGS.max_iters) and step != 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)

train()
