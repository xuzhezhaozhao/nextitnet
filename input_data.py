#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from collections import Counter


class InputData(object):

    def __init__(
            self,
            train_data_path,
            eval_data_path,
            model_dir,
            min_count,
            max_seq_length,
            batch_size,
            eval_batch_size,
            epoch,
            shuffle,
            num_parallel_calls
    ):
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.model_dir = model_dir
        self.min_count = min_count
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.epoch = epoch
        self.shuffle = shuffle
        self.num_parallel_calls = num_parallel_calls

        self.keys_path = os.path.join(self.model_dir, 'keys.dict')

        self.build_vocabulary()

    def build_vocabulary(self):
        data_path = self.train_data_path
        min_count = self.min_count
        counter = Counter()
        cnt = 0
        for line in open(data_path):
            cnt += 1
            tokens = line.split()
            for token in tokens:
                if token.startswith('__label__'):
                    continue
                counter[token] += 1
        self.num_train_samples = cnt

        self.word_to_id = {}
        self.vocab = [""]  # one padding
        self.freqs = [0]
        idx = 1
        for key, cnt in counter.most_common():
            if cnt < min_count:
                continue
            self.word_to_id[key] = idx
            self.vocab.append(key)
            self.freqs.append(cnt)
            idx += 1
        self.vocabulary_size = len(self.vocab)
        tf.logging.info("**** Vocabulary Info ****")
        tf.logging.info(" vocabulary size = %d", self.vocabulary_size)
        tf.logging.info(" key[1] = %s", self.vocab[1])
        tf.logging.info(" key[2] = %s", self.vocab[2])
        tf.logging.info(" key[3] = %s", self.vocab[3])
        tf.logging.info(" key[4] = %s", self.vocab[4])
        tf.logging.info(" key[5] = %s", self.vocab[5])
        tf.logging.info(" key[6] = %s", self.vocab[6])
        tf.logging.info(" key[7] = %s", self.vocab[7])
        tf.logging.info(" key[8] = %s", self.vocab[8])
        tf.logging.info(" key[9] = %s", self.vocab[9])
        tf.logging.info(" key[10] = %s", self.vocab[10])

        # write vocabulary to file
        with open(self.keys_path, 'w') as f:
            for key, cnt in zip(self.vocab, self.freqs):
                if key == "":
                    continue
                f.write(key + ' ' + str(cnt) + '\n')

    def build_train_samples(self):
        features = []
        labels = []
        for line in open(self.train_data_path):
            feature, label = self.parse_line(line)
            features.append(feature)
            labels.append(label)
        self.features = np.array(features)
        self.labels = np.array(labels)

    def build_eval_samples(self):
        features = []
        labels = []
        for line in open(self.eval_data_path):
            feature, label = self.parse_line(line)
            features.append(feature)
            labels.append(label)
        self.eval_features = np.array(features)
        self.eval_labels = np.array(labels)
        self.num_eval_samples = len(self.eval_features)

    def build_numpy_train_input_fn(self):
        self.build_train_samples()
        return tf.estimator.inputs.numpy_input_fn(
            x={'input_ids': self.features},
            y=self.labels,
            batch_size=self.batch_size,
            num_epochs=self.epoch,
            shuffle=self.shuffle)

    def build_numpy_eval_input_fn(self):
        self.build_eval_samples()
        return tf.estimator.inputs.numpy_input_fn(
            x={'input_ids': self.eval_features},
            y=self.eval_labels,
            batch_size=self.eval_batch_size,
            num_epochs=1,
            shuffle=False)

    # Use tf.data
    def parse_py_function(self, line):
        feature, label = self.parse_line(line)
        return feature, label

    def py_wrapper(self, line):
        feature, label = tf.py_func(
            self.parse_py_function, [line], [tf.int64, tf.int64])
        feature.set_shape([self.max_seq_length - 1])
        label.set_shape([self.max_seq_length - 1])
        return {'input_ids': feature}, label

    def build_ds_train_input_fn(self):
        def train_input_fn():
            ds = tf.data.TextLineDataset(self.train_data_path)
            ds = ds.map(
                self.py_wrapper,
                num_parallel_calls=self.num_parallel_calls)
            ds.prefetch(100)
            if self.shuffle:
                ds = ds.shuffle(buffer_size=1000)
            ds = ds.batch(self.batch_size)
            ds = ds.repeat(self.epoch)
            return ds

        return train_input_fn

    def build_ds_eval_input_fn(self):
        def eval_input_fn():
            ds = tf.data.TextLineDataset(self.eval_data_path)
            ds = ds.map(
                self.py_wrapper,
                num_parallel_calls=self.num_parallel_calls)
            ds.prefetch(100)
            ds = ds.batch(self.eval_batch_size)
            return ds
        return eval_input_fn

    def parse_line(self, line):
        tokens = line.split()
        ids = []
        for token in tokens:
            if token.startswith('__label__'):
                continue
            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
            # skip OOV words
        if len(ids) < 2:
            length = self.max_seq_length - 1
            paddings = [0] * length
            return paddings, paddings

        ids = ids[-self.max_seq_length:]
        if len(ids) < self.max_seq_length:
            for _ in range(self.max_seq_length - len(ids)):
                ids.insert(0, 0)
        return ids[:-1], ids[1:]

    def build_serving_input_fn(self):
        def serving_input_receiver_fn():
            feature_spec = {
                'input_ids': tf.FixedLenFeature(
                    shape=[self.max_seq_length - 1], dtype=tf.int64)
            }
            serialized_tf_example = tf.placeholder(
                dtype=tf.string, shape=[None])
            receiver_tensors = {'examples': serialized_tf_example}
            features = tf.parse_example(serialized_tf_example, feature_spec)
            return tf.estimator.export.ServingInputReceiver(
                features, receiver_tensors)

        return serving_input_receiver_fn
