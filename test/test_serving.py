#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import grpc
import tensorflow as tf
import tensorflow_serving.apis.predict_pb2 as predict_pb2
import tensorflow_serving.apis.prediction_service_pb2_grpc as prediction_service_pb2_grpc

MODEL_NAME = "nextitem"


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def run():
    channel = grpc.insecure_channel('127.0.0.1:9000')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME

    request.model_spec.signature_name = 'recall'
    input_name = 'examples'
    inputs = ['4395cbace39695aw']
    to_ranking_inputs = ['8395cc37ce8658bk', '', '6565cbd3193976aw', '2235cbe9d7a891ah']
    example1 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'inputs': _bytes_feature(inputs),
                'to_ranking_inputs': _bytes_feature(to_ranking_inputs)
            }
        )
    ).SerializeToString()

    print("example len = {}".format(len(example1)))
    examples = [example1]
    request.inputs[input_name].CopyFrom(
        tf.contrib.util.make_tensor_proto(examples, dtype=tf.string))

    response = stub.Predict(request)
    print("Received: \n{}".format(response))


if __name__ == '__main__':
    run()
