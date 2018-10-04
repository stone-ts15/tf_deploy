# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Segmentation results visualization on a given set of images.

See model.py for more details and usage.
"""
import math
import os.path
import time
import numpy as np
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import save_annotation

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for visualizing the model.

flags.DEFINE_integer('vis_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('vis_crop_size', [513, 513],
                           'Crop size [height, width] for visualization.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

# Dataset settings.

flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('vis_split', 'val',
                    'Which split of the dataset used for visualizing results')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_enum('colormap_type', 'pascal', ['pascal', 'cityscapes', 'homeplus'],
                  'Visualization colormap type.')

flags.DEFINE_boolean('also_save_raw_predictions', False,
                     'Also save raw predictions.')

flags.DEFINE_integer('max_number_of_iterations', 0,
                     'Maximum number of visualization iterations. Will loop '
                     'indefinitely upon nonpositive values.')

# The folder where semantic segmentation predictions are saved.
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'

# The folder where raw semantic segmentation predictions are saved.
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw_segmentation_results'

# The format to save image.
# _IMAGE_FORMAT = '%06d_image'
_IMAGE_FORMAT = '{}_image'

# The format to save prediction
# _PREDICTION_FORMAT = '%06d_prediction'
_PREDICTION_FORMAT = '{}_prediction'

# To evaluate Cityscapes results on the evaluation server, the labels used
# during training should be mapped to the labels for evaluation.
_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]


def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
  """Converts the predicted label for evaluation.

  There are cases where the training labels are not equal to the evaluation
  labels. This function is used to perform the conversion so that we could
  evaluate the results on the evaluation server.

  Args:
    prediction: Semantic segmentation prediction.
    train_id_to_eval_id: A list mapping from train id to evaluation id.

  Returns:
    Semantic segmentation prediction whose labels have been changed.
  """
  converted_prediction = prediction.copy()
  for train_id, eval_id in enumerate(train_id_to_eval_id):
    converted_prediction[prediction == train_id] = eval_id

  return converted_prediction


def _process_batch(sess, original_images, semantic_predictions, image_names,
                   image_heights, image_widths, image_id_offset, save_dir,
                   raw_save_dir, train_id_to_eval_id=None):
  """Evaluates one single batch qualitatively.

  Args:
    sess: TensorFlow session.
    original_images: One batch of original images.
    semantic_predictions: One batch of semantic segmentation predictions.
    image_names: Image names.
    image_heights: Image heights.
    image_widths: Image widths.
    image_id_offset: Image id offset for indexing images.
    save_dir: The directory where the predictions will be saved.
    raw_save_dir: The directory where the raw predictions will be saved.
    train_id_to_eval_id: A list mapping from train id to eval id.
  """
  (original_images,
   semantic_predictions,
   image_names,
   image_heights,
   image_widths) = sess.run([original_images, semantic_predictions,
                             image_names, image_heights, image_widths])

  num_image = semantic_predictions.shape[0]
  input('num_image: %d' % num_image)
  for i in range(num_image):
    image_height = np.squeeze(image_heights[i])
    image_width = np.squeeze(image_widths[i])
    original_image = np.squeeze(original_images[i])
    semantic_prediction = np.squeeze(semantic_predictions[i])
    crop_semantic_prediction = semantic_prediction[:image_height, :image_width]

    input('save_dir: %s' % save_dir)
    # Save image.
    print(image_names[i].decode("utf-8"))
    save_annotation.save_annotation(
        original_image, save_dir, _IMAGE_FORMAT.format(image_names[i].decode("utf-8")),
        add_colormap=False)

    # Save prediction.
    save_annotation.save_annotation(
        crop_semantic_prediction, save_dir,
        _PREDICTION_FORMAT.format(image_names[i].decode("utf-8")), add_colormap=True,
        colormap_type=FLAGS.colormap_type)

    # if FLAGS.also_save_raw_predictions:
    #   input('also_save_raw')
    #   image_filename = os.path.basename(image_names[i])
    #
    #   if train_id_to_eval_id is not None:
    #     crop_semantic_prediction = _convert_train_id_to_eval_id(
    #         crop_semantic_prediction,
    #         train_id_to_eval_id)
    #   save_annotation.save_annotation(
    #       crop_semantic_prediction, raw_save_dir, image_filename,
    #       add_colormap=False)


def main(unused_argv):
  # input('in vis.main:')
  tf.logging.set_verbosity(tf.logging.INFO)
  # Get dataset-dependent information.
  # FLAGS.dataset: homeplus
  # FLAGS.vis_split: val
  # FLAGS.dataset_dir: tfrecord
  dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset, FLAGS.vis_split, dataset_dir=FLAGS.dataset_dir)

  # input('dataset finish')
  train_id_to_eval_id = None
  # if dataset.name == segmentation_dataset.get_cityscapes_dataset_name():
  #   tf.logging.info('Cityscapes requires converting train_id to eval_id.')
  #   train_id_to_eval_id = _CITYSCAPES_TRAIN_ID_TO_EVAL_ID

  # Prepare for visualization.
  tf.gfile.MakeDirs(FLAGS.vis_logdir)
  save_dir = os.path.join(FLAGS.vis_logdir, _SEMANTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(save_dir)
  raw_save_dir = os.path.join(
      FLAGS.vis_logdir, _RAW_SEMANTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(raw_save_dir)

  tf.logging.info('Visualizing on %s set', FLAGS.vis_split)

  input('prepare finished')

  g = tf.Graph()
  with g.as_default():
    # get preprocessed data
    print('min_resize_value: %s' % str(FLAGS.min_resize_value))
    print('max_resize_value: %s' % str(FLAGS.max_resize_value))
    input('--')
    print(FLAGS.vis_crop_size)
    print(FLAGS.vis_batch_size)
    print(FLAGS.min_resize_value)
    print(FLAGS.max_resize_value)
    print(FLAGS.resize_factor)
    print(FLAGS.vis_split)
    print(FLAGS.model_variant)
    print(FLAGS.atrous_rates)
    print(FLAGS.output_stride)
    # print(FLAGS)
    samples = input_generator.get(dataset,                                  # a dataset
                                  FLAGS.vis_crop_size,                      # [1505, 2049]
                                  FLAGS.vis_batch_size,                     # 1
                                  min_resize_value=FLAGS.min_resize_value,  # None
                                  max_resize_value=FLAGS.max_resize_value,  # None
                                  resize_factor=FLAGS.resize_factor,
                                  dataset_split=FLAGS.vis_split,            # val
                                  is_training=False,
                                  model_variant=FLAGS.model_variant)        # xception 65

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=FLAGS.vis_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)
    print(samples[common.IMAGE])

    input('before predict')
    # maybe predict
    print(dataset.num_classes)
    print(FLAGS.vis_crop_size)
    print(FLAGS.atrous_rates)
    print(FLAGS.output_stride)
    input()
    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      # images: A tensor of size [batch, height, width, channels]
      predictions = model.predict_labels(
          samples[common.IMAGE],
          model_options=model_options,
          image_pyramid=FLAGS.image_pyramid)
    predictions = predictions[common.OUTPUT_TYPE]
    print(predictions.shape)
    print(predictions.dtype)
    input()

    input('predictions finish')
    if FLAGS.min_resize_value and FLAGS.max_resize_value:
      input('not pos')
      # Only support batch_size = 1, since we assume the dimensions of original
      # image after tf.squeeze is [height, width, 3].
      assert FLAGS.vis_batch_size == 1

      # Reverse the resizing and padding operations performed in preprocessing.
      # First, we slice the valid regions (i.e., remove padded region) and then
      # we reisze the predictions back.
      original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
      original_image_shape = tf.shape(original_image)
      predictions = tf.slice(
          predictions,
          [0, 0, 0],
          [1, original_image_shape[0], original_image_shape[1]])
      resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
                                   tf.squeeze(samples[common.WIDTH])])
      predictions = tf.squeeze(
          tf.image.resize_images(tf.expand_dims(predictions, 3),
                                 resized_shape,
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                 align_corners=True), 3)
    input('before saver')
    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(slim.get_variables_to_restore())
    sv = tf.train.Supervisor(graph=g,
                             logdir=FLAGS.vis_logdir,
                             init_op=tf.global_variables_initializer(),
                             summary_op=None,
                             summary_writer=None,
                             global_step=None,
                             saver=saver)

    print('dataset.num_samples: ' + str(dataset.num_samples))
    print('FLAGS.vis_batch_size: ' + str(FLAGS.vis_batch_size))
    # input()
    num_batches = int(math.ceil(dataset.num_samples / float(FLAGS.vis_batch_size)))
    last_checkpoint = None

    # Loop to visualize the results when new checkpoint is created.
    # last_checkpoint = slim.evaluation.wait_for_new_checkpoint(FLAGS.checkpoint_dir, last_checkpoint)
    tf.logging.info(
          'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
    tf.logging.info('Visualizing with model %s', last_checkpoint)

    # start nvidia in opening session
    with sv.managed_session(FLAGS.master, start_standard_services=False) as sess:
      input('in session')
      print(FLAGS.checkpoint_dir)
      sv.start_queue_runners(sess)
      my_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
      sv.saver.restore(sess, my_checkpoint)
      # sv.saver.restore(sess, last_checkpoint)

      tf.logging.info('Visualizing batch %d / %d', 1, num_batches)

      input('before batch')
      # save one prediction png
      _process_batch(sess=sess,
                         original_images=samples[common.ORIGINAL_IMAGE],
                         semantic_predictions=predictions,
                         image_names=samples[common.IMAGE_NAME],
                         image_heights=samples[common.HEIGHT],
                         image_widths=samples[common.WIDTH],
                         image_id_offset=0,
                         save_dir=save_dir,
                         raw_save_dir=raw_save_dir,
                         train_id_to_eval_id=train_id_to_eval_id)


      tf.logging.info(
          'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
      # hack: only conduct visulization once

    # while (FLAGS.max_number_of_iterations <= 0 or
    #        num_iters < FLAGS.max_number_of_iterations):
    #   if num_iters > 0:
    #     time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
    #     if time_to_next_eval > 0:
    #       time.sleep(time_to_next_eval)
    #   num_iters += 1
    #   last_checkpoint = slim.evaluation.wait_for_new_checkpoint(
    #       FLAGS.checkpoint_dir, last_checkpoint)
    #   start = time.time()
    #   tf.logging.info(
    #       'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
    #                                                    time.gmtime()))
    #   tf.logging.info('Visualizing with model %s', last_checkpoint)
    #
    #   with sv.managed_session(FLAGS.master,
    #                           start_standard_services=False) as sess:
    #     sv.start_queue_runners(sess)
    #     sv.saver.restore(sess, last_checkpoint)
    #
    #     image_id_offset = 0
    #     for batch in range(num_batches):
    #       tf.logging.info('Visualizing batch %d / %d', batch + 1, num_batches)
    #
    #       input('before batch')
    #       _process_batch(sess=sess,
    #                      original_images=samples[common.ORIGINAL_IMAGE],
    #                      semantic_predictions=predictions,
    #                      image_names=samples[common.IMAGE_NAME],
    #                      image_heights=samples[common.HEIGHT],
    #                      image_widths=samples[common.WIDTH],
    #                      image_id_offset=image_id_offset,
    #                      save_dir=save_dir,
    #                      raw_save_dir=raw_save_dir,
    #                      train_id_to_eval_id=train_id_to_eval_id)
    #
    #       image_id_offset += FLAGS.vis_batch_size
    #
    #   tf.logging.info(
    #       'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
    #                                                    time.gmtime()))
    #   # hack: only conduct visulization once
    #   break


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('vis_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
