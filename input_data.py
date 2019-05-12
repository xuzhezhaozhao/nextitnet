#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from collections import Counter


class InputData(object):

    def __init__(self, flags):
        self.flags = flags
        self.word_to_id, self.id_to_word = self.build_vocabulary()

    def build_vocabulary(self):
        data_path = self.flags.train_data_path
        min_count = self.flags.min_count
        counter = Counter()
        for line in open(data_path):
            tokens = line.split()
            for token in tokens:
                if token.startswith('__label__'):
                    continue
                counter[token] += 1

        word_to_id = {}
        id_to_word = [""]  # one padding
        idx = 1
        for key in counter:
            cnt = counter[key]
            if cnt < min_count:
                continue
            word_to_id[key] = idx
            id_to_word.append(key)
            idx += 1
        tf.logging.info("**** Vocabulary Info ****")
        tf.logging.info(" vocabulary size = %d", len(id_to_word))

        return word_to_id, id_to_word

    def build_train_input_fn(self):
        features = []
        labels = []
        for line in open(self.flags.train_data_path):
            tokens = line.split()
            ids = []
            for token in tokens:
                if token.startswith('__label__'):
                    continue
                if token in self.word_to_id:
                    ids.append(self.word_to_id[token])
                # TODO how to handle OOV?
            if len(ids) < 2:
                continue
            ids = ids[-self.flags.max_seq_lengh:]
            if len(ids) < self.flags.max_seq_lengh:
                for _ in range(self.flags.max_seq_lengh - len(ids)):
                    ids.append(0)
            features.append(ids[:-1])
            labels.append(ids[1:])
        features = np.array(features)
        labels = np.array(labels)
        tf.logging.info("**** Samples Info ****")
        tf.logging.info(" samples num = %d", len(features))

        return tf.estimator.inputs.numpy_input_fn(
            x={'input_ids': features},
            y=labels,
            batch_size=self.flags.batch_size,
            num_epochs=self.flags.epoch,
            shuffle=True
        )
