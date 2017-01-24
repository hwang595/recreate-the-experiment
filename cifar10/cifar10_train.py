# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import numpy as np
import os.path
import random

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

def calc_training_error(train_set, train_labels, scope, batch_size, sess=None):
  scope.reuse_variables()
  true_count_train = 0
  idx = 0
  for i in xrange(int(train_set.get_shape()[0].value / batch_size)):
    if i % 10 == 0:
      print(i)
    sub_set_train = tf.slice(train_set, [idx, 0, 0, 0], [batch_size, -1, -1, -1])
    sub_labels_train = tf.slice(train_labels, [idx], [batch_size])
    logits_train_err = cifar10.inference(sub_set_train)
    top_k_op_train = tf.nn.in_top_k(logits_train_err, sub_labels_train, 1)
    predictions_train = sess.run([top_k_op_train])
    true_count_train += np.sum(predictions_train)
    idx += batch_size
  train_err = (int(train_set.get_shape()[0]) - true_count_train) / float(train_set.get_shape()[0].value)
  return train_err

def calc_test_error(test_set, test_labels, scope, batch_size, sess=None):
  scope.reuse_variables()
  true_count_test = 0
  idy = 0
  print('======================================================================================================')
  for i in xrange(int(test_set.get_shape()[0].value / batch_size)):
    if i % 10 == 0:
      print(i)
    sub_set_test = tf.slice(test_set, [idy, 0, 0, 0], [batch_size, -1, -1, -1])
    sub_labels_test = tf.slice(test_labels, [idy], [batch_size])
    logits_test_err = cifar10.inference(sub_set_test)
    top_k_op_test = tf.nn.in_top_k(logits_test_err, sub_labels_test, 1)
    predictions_test = sess.run([top_k_op_test])
    true_count_test += np.sum(predictions_test)
    idy += batch_size
  test_err = (int(test_set.get_shape()[0]) - true_count_test) / float(test_set.get_shape()[0].value)
  return test_err

def calc_test_loss(test_set, test_labels, scope, batch_size, sess=None):
  scope.reuse_variables()
  true_count_test = 0
  loss_test_list = []
  idy = 0
  print('======================================================================================================')
  for i in xrange(int(test_set.get_shape()[0].value / batch_size)):
    if i % 10 == 0:
      print(i)
    sub_set_test = tf.slice(test_set, [idy, 0, 0, 0], [batch_size, -1, -1, -1])
    sub_labels_test = tf.slice(test_labels, [idy], [batch_size])
    logits_test_loss = cifar10.inference(sub_set_test)
    loss_test = cifar10.loss(logits_test_loss, sub_labels_test)
    loss_test_val = sess.run(loss_test)
    loss_test_list.append(loss_test_val)
    idy += batch_size
  return sum(loss_test_list) / float(len(loss_test_list))

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    with tf.variable_scope("recover_paper_experiments") as scope:
      global_step = tf.contrib.framework.get_or_create_global_step()
      saver = tf.train.Saver(tf.global_variables())

      # Get images and labels for CIFAR-10.
      images, labels = cifar10.inputs(False)
      train_set, train_labels = cifar10.train_set()
      test_set, test_labels = cifar10.test_set()

      # Build a Graph that computes the logits predictions from the
      # inference model.
      logits_train = cifar10.inference(images)

      # Calculate loss.
      loss = cifar10.loss(logits_train, labels)

      # Build a Graph that trains the model with one batch of examples and
      # updates the model parameters.
      train_op = cifar10.train(loss, global_step)

      init = tf.global_variables_initializer()

      # Start running operations on the Graph. allow_soft_placement must be set to
      # True to build towers on GPU, as some of the ops do not have GPU
      # implementations.
      sess = tf.Session(config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement))
      sess.run(init)

      # Start the queue runners.
      tf.train.start_queue_runners(sess=sess)

#      summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

      for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))

        if step % 390 == 0:
          if step / 390 > 15:
   #       summary_str = sess.run(summary_op)
            train_err = calc_training_error(
              train_set, train_labels, scope, FLAGS.batch_size, sess=sess)
            test_err = calc_test_error(
              test_set, test_labels, scope, FLAGS.batch_size, sess=sess)
            format_str_per_epoch_train = ('epoch: %s, train err: %.4f')
            format_str_per_epoch_test = ('epoch: %s, test err: %.4f')
  #          test_err = calc_test_error(
  #            test_set, test_labels, scope, sess=sess)
            print (format_str_per_epoch_train % (step / 390, train_err))
            print (format_str_per_epoch_test % (step / 390, test_err))
            with open("log_abs_err_test.dat", "a") as log_file:
              log_file.write("%d, %.4f, %.4f\n" % (step / 390, train_err, test_err))
  #        summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
    '''
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
        if self._step % 390 == 0: #which means we run a epoch for the global process
          train_err = calc_training_error(train_set)
          format_str_per_epoch = ('epoch: %s, train err: %.2f')
          print (format_str_per_epoch % (self._step / 390, train_err))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
    '''


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
