
import numpy as np
import math
import os
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator, save_annotation
from deeplab import common, model
import tensorflow as tf

program_dir = '/home/chr/sgy/web/program'
work_dir = os.path.join(program_dir, 'assets', 'datasets')

A_dataset = 'homeplus'
A_vis_split = 'val'
A_dataset_dir = os.path.join(work_dir, 'tfrecord')

A_min_resize_value = None
A_max_resize_value = None
A_resize_factor = None
A_vis_crop_size = [1505, 2049]
A_vis_batch_size = 1
A_model_variant = 'xception_65'
A_atrous_rates = [6, 12, 18]
A_output_stride = 16
A_image_pyramid = None
A_colormap_type = 'pascal'

_IMAGE_FORMAT = '{}_image'
_PREDICTION_FORMAT = '{}_prediction'
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'


def _process_batch(sess, original_images, semantic_predictions, image_names,
                   image_heights, image_widths, save_dir):
    (original_images,
     semantic_predictions,
     image_names,
     image_heights,
     image_widths) = sess.run([original_images, semantic_predictions,
                               image_names, image_heights, image_widths])

    num_image = semantic_predictions.shape[0]
    i = 0
    image_height = np.squeeze(image_heights[i])
    image_width = np.squeeze(image_widths[i])
    original_image = np.squeeze(original_images[i])
    semantic_prediction = np.squeeze(semantic_predictions[i])
    crop_semantic_prediction = semantic_prediction[:image_height, :image_width]

    save_annotation.save_annotation(
        original_image, save_dir, _IMAGE_FORMAT.format(image_names[i].decode("utf-8")),
        add_colormap=False)

    save_annotation.save_annotation(
        crop_semantic_prediction, save_dir,
        _PREDICTION_FORMAT.format(image_names[i].decode("utf-8")), add_colormap=True,
        colormap_type=A_colormap_type)


def vis_main(sess):
    dataset = segmentation_dataset.get_dataset(A_dataset, A_vis_split, dataset_dir=A_dataset_dir)

    train_id_to_eval_id = None

    save_dir = os.path.join(work_dir, _SEMANTIC_PREDICTION_SAVE_FOLDER)

    print('min_resize_value: %s' % str(A_min_resize_value))
    print('max_resize_value: %s' % str(A_max_resize_value))

    samples = input_generator.get(dataset,
                                  A_vis_crop_size,
                                  A_vis_batch_size,
                                  min_resize_value=A_min_resize_value,
                                  max_resize_value=A_max_resize_value,
                                  resize_factor=A_resize_factor,
                                  dataset_split=A_vis_split,
                                  is_training=False,
                                  model_variant=A_model_variant)

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=A_vis_crop_size,
        atrous_rates=A_atrous_rates,
        output_stride=A_output_stride
    )

    print(samples[common.IMAGE])

    predictions = model.predict_labels(samples[common.IMAGE],
                                       model_options=model_options,
                                       image_pyramid=A_image_pyramid)

    predictions = predictions[common.OUTPUT_TYPE]

    # _process_batch(sess,
    #                samples[common.ORIGINAL_IMAGE],
    #                predictions,
    #                samples[common.IMAGE_NAME],
    #                samples[common.HEIGHT],
    #                samples[common.WIDTH],
    #                save_dir)

def do_process_batch(sess, samples, predictions):
    save_dir = os.path.join(work_dir, _SEMANTIC_PREDICTION_SAVE_FOLDER)

    _process_batch(sess,
                   samples[common.ORIGINAL_IMAGE],
                   predictions,
                   samples[common.IMAGE_NAME],
                   samples[common.HEIGHT],
                   samples[common.WIDTH],
                   save_dir)

def do_prepare():
    dataset = segmentation_dataset.get_dataset(A_dataset, A_vis_split, dataset_dir=A_dataset_dir)
    # with tf.Graph().as_default():
    # with tf.get_default_graph():
    samples = input_generator.get(dataset,
                                  A_vis_crop_size,
                                  A_vis_batch_size,
                                  min_resize_value=A_min_resize_value,
                                  max_resize_value=A_max_resize_value,
                                  resize_factor=A_resize_factor,
                                  dataset_split=A_vis_split,
                                  is_training=False,
                                  model_variant=A_model_variant)

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=A_vis_crop_size,
        atrous_rates=A_atrous_rates,
        output_stride=A_output_stride
    )
    print(samples[common.IMAGE])

    predictions = model.predict_labels(samples[common.IMAGE],
                                       model_options=model_options,
                                       image_pyramid=A_image_pyramid)

    predictions = predictions[common.OUTPUT_TYPE]

    return samples, predictions