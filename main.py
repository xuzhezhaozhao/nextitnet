#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import modeling
import input_data


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'

flags = tf.flags.FLAGS


tf.app.flags.DEFINE_string('model_dir', 'model_dir', '')
tf.app.flags.DEFINE_string('export_model_dir', 'export_model_dir', '')
tf.app.flags.DEFINE_bool('do_train', False, '')
tf.app.flags.DEFINE_bool('do_eval', False, '')
tf.app.flags.DEFINE_bool('do_export', False, '')
tf.app.flags.DEFINE_string('export_mode', 'recall', 'recall or ranking')
tf.app.flags.DEFINE_string('train_data_path', '', 'train data path')
tf.app.flags.DEFINE_string('eval_data_path', '', 'eval data path')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('eval_batch_size', 64, 'eval batch size')
tf.app.flags.DEFINE_integer('epoch', 5, '')
tf.app.flags.DEFINE_integer('min_count', 5, '')
tf.app.flags.DEFINE_integer('max_seq_length', 5, '')
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
tf.app.flags.DEFINE_integer('num_parallel_calls', 1, '')

# log flags
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_integer('save_checkpoints_steps', 100, '')
tf.app.flags.DEFINE_integer('keep_checkpoint_max', 3, '')
tf.app.flags.DEFINE_integer('log_step_count_steps', 100, '')


def build_estimator(
        flags, num_classes, num_train_steps, num_warmup_steps, unigrams, keys):
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
        flags.recall_k,
        unigrams,
        keys,
        flags.batch_size,
        flags.max_seq_length,
        flags.model_dir
    )
    estimator_keys['config'] = config
    estimator = tf.estimator.Estimator(**estimator_keys)

    return estimator


def model_fn_builder(num_classes, embedding_dim, dilations, kernel_size,
                     num_sampled, learning_rate,
                     num_train_steps, num_warmup_steps, recall_k,
                     unigrams, keys, batch_size, max_seq_length, model_dir):
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
        if mode == tf.estimator.ModeKeys.TRAIN:
            labels = tf.reshape(labels, [-1, 1])
            valid_idx = tf.where(tf.not_equal(labels, 0))[:, 0]

            loss = optimized_nce_loss(
                model.nce_weights,
                model.nce_biases,
                labels,
                model.output_2d,
                unigrams,
                num_sampled,
                num_classes,
                embedding_dim,
                valid_idx,
                batch_size,
                max_seq_length
            )
            loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss', loss)

            if flags.num_gpu > 1:
                import optimization_multi_gpu as optimization
            else:
                import optimization as optimization

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
            labels = tf.reshape(labels, [-1, 1])
            valid_idx = tf.where(tf.not_equal(labels, 0))[:, 0]
            labels = tf.nn.embedding_lookup(labels, valid_idx)
            output = tf.nn.embedding_lookup(model.output_2d, valid_idx)
            logits = tf.nn.xw_plus_b(
                output, tf.transpose(model.nce_weights), model.nce_biases)
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
            output = model.output_3d[:, -1:, :]   # sequence predict
            output = tf.reshape(output, [-1, embedding_dim])
            nce_weights = np.load(os.path.join(model_dir, 'nce_weights.npy'))
            nce_biases = np.load(os.path.join(model_dir, 'nce_biases.npy'))
            nce_weights_transpose = tf.convert_to_tensor(
                nce_weights.transpose(), dtype=tf.float32)
            nce_biases_transpose = tf.convert_to_tensor(
                nce_biases.transpose(), dtype=tf.float32)
            logits = tf.nn.xw_plus_b(
                output, nce_weights_transpose, nce_biases_transpose)
            probs = tf.nn.softmax(logits)
            scores, ids = tf.nn.top_k(probs, recall_k)
            table = tf.contrib.lookup.index_to_string_table_from_tensor(
                mapping=keys,
                default_value='')
            outputs = {
                'scores': scores,
                'keys': table.lookup(tf.cast(ids, tf.int64)),
            }
            export_outputs = {
                'predicts': tf.estimator.export.PredictOutput(outputs=outputs),
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode,
                predictions=outputs,
                export_outputs=export_outputs)
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

        metrics = {
            'accuracy': accuracy,
            'recall_at_top_{}'.format(recall_k): recall_at_top_k,
            'recall_at_top_{}'.format(recall_k2): recall_at_top_k2,
            'recall_at_top_{}'.format(recall_k4): recall_at_top_k4
        }
        for key in metrics.keys():
            tf.summary.scalar(key, metrics[key][1])
    return metrics


def optimized_nce_loss(weights, biases, labels, inputs, unigrams, num_sampled,
                       num_classes, nce_dim, valid_idx, batch_size,
                       max_seq_length):

    batch_size = batch_size * (max_seq_length - 1)

    # [batch, num_sampled]
    sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels,
        num_true=1,
        num_sampled=num_sampled * batch_size,
        unique=False,
        range_max=num_classes,
        distortion=1.0,
        num_reserved_ids=0,
        num_shards=1,
        shard=0,
        unigrams=unigrams,
        seed=None,
        name=None
    )
    sampled_ids = tf.reshape(sampled_ids, [batch_size, num_sampled])

    # Weights for labels: [batch_size, emb_dim]
    true_w = tf.nn.embedding_lookup(weights, tf.reshape(labels, [-1]))
    # Biases for labels: [batch_size, emb_dim]
    true_b = tf.nn.embedding_lookup(biases, tf.reshape(labels, [-1]))

    # Weights for sampled ids: [batch, num_sampled, emb_dim]
    sampled_w = tf.nn.embedding_lookup(weights, sampled_ids)
    # Biases for sampled ids: [batch, num_sampled, 1]
    sampled_b = tf.nn.embedding_lookup(biases, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.multiply(inputs, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    sampled_b_vec = tf.reshape(sampled_b, [-1, num_sampled])
    broadcast_inputs = tf.reshape(inputs, [-1, 1, nce_dim])
    sampled_logits = tf.multiply(broadcast_inputs, sampled_w)
    sampled_logits = tf.reduce_sum(sampled_logits, -1) + sampled_b_vec

    # cross-entropy(logits, labels)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

    # loss mask
    true_xent = tf.nn.embedding_lookup(true_xent, valid_idx)
    sampled_xent = tf.nn.embedding_lookup(sampled_xent, valid_idx)

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (tf.reduce_sum(true_xent) +
                       tf.reduce_sum(sampled_xent)) / batch_size
    return nce_loss_tensor


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    data = input_data.InputData(
        train_data_path=flags.train_data_path,
        eval_data_path=flags.eval_data_path,
        model_dir=flags.model_dir,
        min_count=flags.min_count,
        max_seq_length=flags.max_seq_length,
        batch_size=flags.batch_size,
        eval_batch_size=flags.eval_batch_size,
        epoch=flags.epoch,
        shuffle=True,
        num_parallel_calls=flags.num_parallel_calls
    )
    num_train_steps = int(data.num_train_samples /
                          flags.batch_size * flags.epoch)
    num_warmup_steps = int(flags.warmup_proportion * num_train_steps)
    estimator = build_estimator(flags, data.vocabulary_size,
                                num_train_steps, num_warmup_steps, data.freqs,
                                data.vocab)
    if flags.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", data.num_train_samples)
        tf.logging.info("  Batch size = %d", flags.batch_size)
        tf.logging.info("  Num train steps = %d", num_train_steps)
        tf.logging.info("  Num warmup steps = %d", num_warmup_steps)
        # estimator.train(input_fn=data.build_numpy_train_input_fn())
        estimator.train(input_fn=data.build_ds_train_input_fn())
        nce_weights = estimator.get_variable_value(
            'nce_layer_variables/nce_weights:0')
        nce_biases = estimator.get_variable_value(
            'nce_layer_variables/nce_biases:0')
        np.save(os.path.join(flags.model_dir, 'nce_weights.npy'), nce_weights)
        np.save(os.path.join(flags.model_dir, 'nce_biases.npy'), nce_biases)
    if flags.do_eval:
        tf.logging.info("***** Running evaluating *****")
        tf.logging.info("  Batch size = %d", flags.eval_batch_size)
        # estimator.evaluate(input_fn=data.build_numpy_eval_input_fn())
        estimator.evaluate(input_fn=data.build_ds_eval_input_fn())
    if flags.do_export:
        tf.logging.info("***** Running exporting *****")
        assets_extra = {}
        assets_extra['keys.dict'] = data.keys_path

        if flags.export_mode == 'recall':
            estimator.export_savedmodel(
                flags.export_model_dir,
                serving_input_receiver_fn=data.build_recall_serving_input_fn(),
                assets_extra=assets_extra)
        elif flags.export_mode == 'ranking':
            estimator.export_savedmodel(
                flags.export_model_dir,
                serving_input_receiver_fn=data.build_ranking_serving_input_fn(),
                assets_extra=assets_extra)
        else:
            raise ValueError("Unknow export mode.")


if __name__ == '__main__':
    tf.app.run()
