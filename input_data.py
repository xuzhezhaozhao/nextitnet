#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from collections import Counter


class InputData(object):

    def __init__(
            self,
            train_data_path,
            eval_data_path,
            min_count,
            max_seq_lengh,
            batch_size,
            eval_batch_size,
            epoch,
            shuffle):
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.min_count = min_count
        self.max_seq_lengh = max_seq_lengh
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.epoch = epoch
        self.shuffle = shuffle

        self.build_vocabulary()
        self.build_train_samples()
        self.build_eval_samples()

    def build_vocabulary(self):
        data_path = self.train_data_path
        min_count = self.min_count
        counter = Counter()
        for line in open(data_path):
            tokens = line.split()
            for token in tokens:
                if token.startswith('__label__'):
                    continue
                counter[token] += 1

        self.word_to_id = {}
        self.id_to_word = [""]  # one padding
        idx = 1
        for key in counter:
            cnt = counter[key]
            if cnt < min_count:
                continue
            self.word_to_id[key] = idx
            self.id_to_word.append(key)
            idx += 1
        self.vocabulary_size = len(self.id_to_word)
        tf.logging.info("**** Vocabulary Info ****")
        tf.logging.info(" vocabulary size = %d", self.vocabulary_size)

    def build_train_samples(self):
        features = []
        labels = []
        for line in open(self.train_data_path):
            tokens = line.split()
            ids = []
            for token in tokens:
                if token.startswith('__label__'):
                    continue
                if token in self.word_to_id:
                    ids.append(self.word_to_id[token])
                # skip OOV words
            if len(ids) < 2:
                continue
            ids = ids[-self.max_seq_lengh:]
            if len(ids) < self.max_seq_lengh:
                for _ in range(self.max_seq_lengh - len(ids)):
                    ids.append(0)
            features.append(ids[:-1])
            labels.append(ids[1:])
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.num_train_samples = len(self.features)

    def build_eval_samples(self):
        features = []
        labels = []
        for line in open(self.eval_data_path):
            tokens = line.split()
            ids = []
            for token in tokens:
                if token.startswith('__label__'):
                    continue
                if token in self.word_to_id:
                    ids.append(self.word_to_id[token])
                else:
                    # TODO whether skip OOV words for eval ?
                    # ids.append(0)
                    pass
            if len(ids) < 2:
                continue
            ids = ids[-self.max_seq_lengh:]
            if len(ids) < self.max_seq_lengh:
                for _ in range(self.max_seq_lengh - len(ids)):
                    ids.append(0)
            features.append(ids[:-1])
            labels.append(ids[1:])
        self.eval_features = np.array(features)
        self.eval_labels = np.array(labels)
        self.num_eval_samples = len(self.eval_features)

    def build_train_input_fn(self):
        return tf.estimator.inputs.numpy_input_fn(
            x={'input_ids': self.features},
            y=self.labels,
            batch_size=self.batch_size,
            num_epochs=self.epoch,
            shuffle=self.shuffle)

    def build_eval_input_fn(self):
        return tf.estimator.inputs.numpy_input_fn(
            x={'input_ids': self.eval_features},
            y=self.eval_labels,
            batch_size=self.eval_batch_size,
            num_epochs=1,
            shuffle=False)
