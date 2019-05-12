#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class NextItemModel(object):

    def __init__(self,
                 input_ids,
                 labels,
                 num_classes,
                 embedding_dim,
                 dilations,
                 kernel_size,
                 num_sampled,
                 training):
        self.input_ids = input_ids
        self.labels = labels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.num_sampled = num_sampled
        self.training = training

        self.build_graph(training=training)

    def build_graph(self, training=True):
        self.embeddings = self.get_embeddings()
        # TODO mask embedding_lookup?
        inputs = tf.nn.embedding_lookup(self.embeddings, self.input_ids)
        for layer_id, dilation in enumerate(self.dilations):
            inputs = self.residual_block(
                inputs, dilation, layer_id, self.embedding_dim,
                self.kernel_size, causal=True, training=training)
        tf.logging.info(" output = %s", inputs)
        logits = tf.reshape(inputs, [-1, self.embedding_dim])
        tf.logging.info(" logits = %s", logits)
        nce_weights, nce_biases = self.get_nce_weights_and_biases()
        labels_flat = tf.reshape(self.labels, [-1, 1])
        loss = tf.nn.sampled_softmax_loss(
            nce_weights, nce_biases, labels_flat, logits,
            self.num_sampled, self.num_classes)

        self.logits = logits
        self.loss = tf.reduce_mean(loss)

    def get_embeddings(self):
        """Get embeddings variables."""

        dim = self.embedding_dim
        init_width = 1.0 / dim
        with tf.variable_scope("embeddings_variable", reuse=tf.AUTO_REUSE):
            embeddings = tf.get_variable(
                "embeddings", initializer=tf.random_uniform(
                    [self.num_classes, dim], -init_width, init_width))
        return embeddings

    def get_nce_weights_and_biases(self):
        """Get nce weights and biases variables."""

        with tf.variable_scope("nce_layer_variables", reuse=tf.AUTO_REUSE):
            nce_weights = tf.get_variable(
                'nce_weights',
                initializer=tf.truncated_normal(
                    [self.num_classes, self.embedding_dim], 0.0, 0.01))
            nce_biases = tf.get_variable(
                'nce_biases',
                initializer=tf.zeros([self.num_classes]))
        return nce_weights, nce_biases

    def residual_block(self, x, dilation, layer_id, channels,
                       kernel_size, causal=True, training=True):
        block_name = "res_{}_{}".format(layer_id, dilation)
        tf.logging.info("build block %s ...", block_name)
        with tf.variable_scope(block_name, reuse=tf.AUTO_REUSE):
            y = self.conv1d(x, channels, dilation, kernel_size,
                            causal=causal, name="dilated_conv1")
            tf.logging.info(y)
            y = self.layer_norm(y, name="layer_norm1", trainable=training)
            tf.logging.info(y)
            y = tf.nn.relu(y)
            tf.logging.info(y)
            y = self.conv1d(y, channels, 2*dilation, kernel_size,
                            causal=causal, name="dilated_conv2")
            tf.logging.info(y)
            y = self.layer_norm(y, name="layer_norm2", trainable=training)
            tf.logging.info(y)
            y = tf.nn.relu(y)
            tf.logging.info(y)
            y += x
            tf.logging.info(y)
            return y

    def conv1d(self, x, channels, dilation=1, kernel_size=1, causal=False,
               name='conv1d'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weight = tf.get_variable(
                'weight',
                [1, kernel_size, x.get_shape()[-1], channels],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            bias = tf.get_variable(
                'bias',
                [channels],
                initializer=tf.constant_initializer(0.0))

            if causal:
                padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
                padded = tf.pad(x, padding)
                input_expanded = tf.expand_dims(padded, axis=1)
                out = tf.nn.atrous_conv2d(input_expanded, weight,
                                          rate=dilation, padding='VALID')
                out += bias
            else:
                input_expanded = tf.expand_dims(x, axis=1)
                out = tf.nn.conv2d(input_expanded, weight,
                                   strides=[1, 1, 1, 1], padding="SAME")
                out += bias

            return tf.squeeze(out, [1])

    def layer_norm(self, x, name, epsilon=1e-8, trainable=True):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            shape = x.get_shape()
            beta = tf.get_variable(
                'beta', [int(shape[-1])],
                initializer=tf.constant_initializer(0),
                trainable=trainable)
            gamma = tf.get_variable(
                'gamma', [int(shape[-1])],
                initializer=tf.constant_initializer(1), trainable=trainable)
            mean, variance = tf.nn.moments(x, axes=[len(shape) - 1],
                                           keep_dims=True)
            x = (x - mean) / tf.sqrt(variance + epsilon)
            return gamma * x + beta
