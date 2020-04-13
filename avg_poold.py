from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numbers
import os

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
# copybara:strip_begin
# TODO(b/138808492): Remove code inside copybara
from tensorflow.python.ops import control_flow_ops
# copybara:strip_end
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation

from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup

from tensorflow.python.util.tf_export import tf_export

# Aliases for some automatically-generated names.
local_response_normalization = gen_nn_ops.lrn

# pylint: disable=protected-access
@tf_export("nn.avg_pool2d", v1=[])
def avg_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):  # pylint: disable=redefined-builtin
  """Performs the average pooling on the input.
  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.
  Args:
    input: A 4-D `Tensor` of shape `[batch, height, width, channels]` and type
      `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
    ksize: An int or list of `ints` that has length `1`, `2` or `4`. The size of
      the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `2` or `4`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the "returns" section of `tf.nn.convolution` for details.
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: Optional name for the operation.
  Returns:
    A `Tensor` with the same type as `value`.  The average pooled output tensor.
  """
  with ops.name_scope(name, "AvgPool2D", [input]) as name:
    if data_format is None:
      data_format = "NHWC"
    channel_index = 1 if data_format.startswith("NC") else 3

    ksize = _get_sequence(ksize, 2, channel_index, "ksize")
    strides = _get_sequence(strides, 2, channel_index, "strides")

    return gen_nn_ops.avg_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)
