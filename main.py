#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import modeling
import optimization
import input_data


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'

flags = tf.flags.FLAGS


tf.app.flags.DEFINE_string('model_dir', 'model_dir', '')
tf.app.flags.DEFINE_string('export_model_dir', 'export_model_dir', '')
tf.app.flags.DEFINE_bool('do_train', False, '')
tf.app.flags.DEFINE_string('train_data_path', '', 'train data path')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('epoch', 5, '')
tf.app.flags.DEFINE_integer('min_count', 1, '')
tf.app.flags.DEFINE_integer('max_seq_lengh', 5, '')
tf.app.flags.DEFINE_integer('embedding_dim', 100, '')
tf.app.flags.DEFINE_list('dilations', '1,2,4', '')
tf.app.flags.DEFINE_integer('kernel_size', 3, '')
tf.app.flags.DEFINE_integer('num_sampled', 10, '')

# log flags
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_integer('save_checkpoints_steps', 100, '')
tf.app.flags.DEFINE_integer('keep_checkpoint_max', 3, '')
tf.app.flags.DEFINE_integer('log_step_count_steps', 100, '')


def build_estimator(flags, num_classes):
    """Build estimator."""

    config_keys = {}
    config_keys['model_dir'] = flags.model_dir
    config_keys['save_summary_steps'] = flags.save_summary_steps
    config_keys['save_checkpoints_steps'] = flags.save_checkpoints_steps
    config_keys['session_config'] = None
    config_keys['keep_checkpoint_max'] = flags.keep_checkpoint_max
    config_keys['log_step_count_steps'] = flags.log_step_count_steps
    config = tf.estimator.RunConfig(**config_keys)
    estimator_keys = {}
    estimator_keys['model_dir'] = flags.model_dir
    dilations = [int(x) for x in flags.dilations]
    estimator_keys['model_fn'] = model_fn_builder(
        num_classes,
        flags.embedding_dim,
        dilations,
        flags.kernel_size,
        flags.num_sampled)
    estimator_keys['config'] = config
    estimator = tf.estimator.Estimator(**estimator_keys)

    return estimator


def model_fn_builder(num_classes, embedding_dim, dilations, kernel_size,
                     num_sampled):
    """Returns 'model_fn' closure for Estimator."""

    def model_fn(features, labels, mode):
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(" name = {}, shape = {}"
                            .format(name, features[name].shape))
        input_ids = features['input_ids']
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.NextItemModel(
            input_ids=input_ids,
            labels=labels,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dilations=dilations,
            kernel_size=kernel_size,
            num_sampled=num_sampled,
            training=is_training)
        loss = model.loss
        tvars = tf.trainable_variables()
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            tf.logging.info(" name = %s, shape = %s", var.name, var.shape)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            # TODO
            train_op = optimization.create_optimizer(
                loss, init_lr=0.01, num_train_steps=1000,
                num_warmup_steps=100, use_tpu=False)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            pass
        else:
            pass
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    data = input_data.InputData(flags)
    estimator = build_estimator(flags, len(data.id_to_word))
    if flags.do_train:
        estimator.train(input_fn=data.build_train_input_fn())


if __name__ == '__main__':
    tf.app.run()
