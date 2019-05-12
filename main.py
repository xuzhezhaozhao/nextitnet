#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'


def build_estimator(flags):
    pass


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == '__main__':
    tf.app.run()
