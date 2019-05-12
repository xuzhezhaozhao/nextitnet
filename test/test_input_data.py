#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from input_data import InputData
from tensorflow.python.training import coordinator


class Flags():
    pass


flags = Flags()
flags.train_data_path = './test/test.txt'
flags.min_count = 1
flags.max_seq_lengh = 5
flags.batch_size = 2
flags.epoch = 1

inputdata = InputData(flags)
features, labels = inputdata.build_train_input_fn()()
print(features['inputs'].get_shape()[-1])
with tf.Session() as sess:
    coord = coordinator.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    print(sess.run([features, labels]))
    print(sess.run([features, labels]))
    coord.request_stop()
    coord.join(threads)
