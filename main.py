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
tf.app.flags.DEFINE_bool('do_eval', False, '')
tf.app.flags.DEFINE_string('train_data_path', '', 'train data path')
tf.app.flags.DEFINE_string('eval_data_path', '', 'eval data path')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('eval_batch_size', 64, 'eval batch size')
tf.app.flags.DEFINE_integer('epoch', 5, '')
tf.app.flags.DEFINE_integer('min_count', 5, '')
tf.app.flags.DEFINE_integer('max_seq_lengh', 5, '')
tf.app.flags.DEFINE_integer('embedding_dim', 100, '')
tf.app.flags.DEFINE_list('dilations', '1,2,4', '')
tf.app.flags.DEFINE_integer('kernel_size', 3, '')
tf.app.flags.DEFINE_integer('num_sampled', 10, '')
tf.app.flags.DEFINE_integer('num_gpu', 0, '')
tf.app.flags.DEFINE_float('learning_rate', 0.025, '')
tf.app.flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")
tf.app.flags.DEFINE_integer('recall_k', 20, '')

# log flags
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_integer('save_checkpoints_steps', 100, '')
tf.app.flags.DEFINE_integer('keep_checkpoint_max', 3, '')
tf.app.flags.DEFINE_integer('log_step_count_steps', 100, '')


def build_estimator(flags, num_classes, num_train_steps, num_warmup_steps):
    """Build estimator."""

    config_keys = {}
    config_keys['model_dir'] = flags.model_dir
    config_keys['save_summary_steps'] = flags.save_summary_steps
    config_keys['save_checkpoints_steps'] = flags.save_checkpoints_steps
    config_keys['session_config'] = None
    config_keys['keep_checkpoint_max'] = flags.keep_checkpoint_max
    config_keys['log_step_count_steps'] = flags.log_step_count_steps

    if flags.num_gpu > 1:
        distribution = tf.contrib.distribute.MirroredStrategy(
            num_gpus=flags.num_gpu)
        config_keys['train_distribute'] = distribution

    config = tf.estimator.RunConfig(**config_keys)
    estimator_keys = {}
    estimator_keys['model_dir'] = flags.model_dir
    dilations = [int(x) for x in flags.dilations]
    estimator_keys['model_fn'] = model_fn_builder(
        num_classes,
        flags.embedding_dim,
        dilations,
        flags.kernel_size,
        flags.num_sampled,
        flags.learning_rate,
        num_train_steps,
        num_warmup_steps,
        flags.recall_k)
    estimator_keys['config'] = config
    estimator = tf.estimator.Estimator(**estimator_keys)

    return estimator


def model_fn_builder(num_classes, embedding_dim, dilations, kernel_size,
                     num_sampled, learning_rate,
                     num_train_steps, num_warmup_steps, recall_k):
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
            training=is_training)

        tvars = tf.trainable_variables()
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            tf.logging.info(" name = %s, shape = %s", var.name, var.shape)

        output_spec = None
        labels = tf.reshape(labels, [-1, 1])
        valid_idx = tf.where(tf.not_equal(labels, 0))[:, 0]
        labels = tf.nn.embedding_lookup(labels, valid_idx)
        if mode == tf.estimator.ModeKeys.TRAIN:
            output = tf.nn.embedding_lookup(model.output, valid_idx)
            loss = tf.nn.sampled_softmax_loss(
                model.nce_weights,
                model.nce_biases,
                labels,
                output,
                num_sampled,
                num_classes,
                partition_strategy="div")
            loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss', loss)
            train_op = optimization.create_optimizer(
                loss=loss,
                init_lr=learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            logits = tf.nn.embedding_lookup(model.logits, valid_idx)
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=tf.reshape(labels, [-1]),
                logits=logits)
            _, ids = tf.nn.top_k(logits, recall_k)
            metrics = create_metrics(labels, logits, ids, recall_k)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=metrics)
        else:
            pass
        return output_spec

    return model_fn


def create_metrics(labels, logits, ids, recall_k):
    """Get metrics dict."""

    recall_k2 = recall_k // 2
    recall_k4 = recall_k // 4
    with tf.name_scope('eval_metrics'):
        predicted = ids[:, :1]
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted)
        recall_at_top_k = tf.metrics.recall_at_top_k(
            labels=labels, predictions_idx=ids, k=recall_k)
        recall_at_top_k2 = tf.metrics.recall_at_top_k(
            labels=labels, predictions_idx=ids[:, :recall_k2], k=recall_k2)
        recall_at_top_k4 = tf.metrics.recall_at_top_k(
            labels=labels, predictions_idx=ids[:, :recall_k4], k=recall_k4)
        mrr_at_k = mrr_metric(labels, logits, recall_k)
        mrr_at_k2 = mrr_metric(labels, logits, recall_k2)
        mrr_at_k4 = mrr_metric(labels, logits, recall_k4)

        metrics = {
            'accuracy': accuracy,
            'recall_at_top_{}'.format(recall_k): recall_at_top_k,
            'recall_at_top_{}'.format(recall_k2): recall_at_top_k2,
            'recall_at_top_{}'.format(recall_k4): recall_at_top_k4,
            'mrr_at_{}'.format(recall_k): mrr_at_k,
            'mrr_at_{}'.format(recall_k2): mrr_at_k2,
            'mrr_at_{}'.format(recall_k4): mrr_at_k4
        }
        for key in metrics.keys():
            tf.summary.scalar(key, metrics[key][1])
    return metrics


def mrr_metric(labels,
               predictions,
               k,
               weights=None,
               metrics_collections=None,
               updates_collections=None,
               name=None):
    with tf.name_scope(name, 'mrr_metric', [predictions, labels, weights]):
        _, r = tf.nn.top_k(predictions, k)
        get_ranked_indicies = tf.expand_dims(tf.where(
            tf.equal(tf.cast(r, tf.int64), labels))[:, 1], 1)
        rr = 1.0 / (tf.to_float(get_ranked_indicies) + 1.0)
        m_rr, update_mrr_op = tf.metrics.mean(rr, weights=weights, name=name)
        if metrics_collections:
            tf.add_to_collection(metrics_collections, m_rr)
        if updates_collections:
            tf.add_to_collections(updates_collections, update_mrr_op)
        return m_rr, update_mrr_op


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    data = input_data.InputData(
        flags.train_data_path,
        flags.eval_data_path,
        flags.min_count,
        flags.max_seq_lengh,
        flags.batch_size,
        flags.eval_batch_size,
        flags.epoch,
        True)
    num_train_steps = int(data.num_train_samples /
                          flags.batch_size * flags.epoch)
    num_warmup_steps = int(flags.warmup_proportion * num_train_steps)
    estimator = build_estimator(flags, data.vocabulary_size,
                                num_train_steps, num_warmup_steps)
    if flags.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", data.num_train_samples)
        tf.logging.info("  Batch size = %d", flags.batch_size)
        tf.logging.info("  Num train steps = %d", num_train_steps)
        tf.logging.info("  Num warmup steps = %d", num_warmup_steps)
        estimator.train(input_fn=data.build_train_input_fn())
    if flags.do_eval:
        tf.logging.info("***** Running evaluating *****")
        tf.logging.info("  Num examples = %d", data.num_eval_samples)
        tf.logging.info("  Batch size = %d", flags.eval_batch_size)
        estimator.evaluate(input_fn=data.build_eval_input_fn())


if __name__ == '__main__':
    tf.app.run()
