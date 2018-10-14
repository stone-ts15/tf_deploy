
import math
import os
import sys
import tensorflow as tf
import build

program_dir = '/home/chr/sgy/web/program'
work_dir = os.path.join(program_dir, 'assets', 'datasets')
work_dir = '/DATA/ylxiong/homeplus/'

A_image_folder = os.path.join(work_dir, 'test_data', 'JPEGImages')
A_list_folder  = os.path.join(work_dir, 'test_data', 'ImageSets', 'Segmentation')
A_output_dir   = os.path.join(work_dir, 'test_tfrecord')
A_image_format = 'jpg'

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('image_folder',
                           os.path.join(work_dir, 'test_data', 'JPEGImages'),
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'semantic_segmentation_folder',
    './VOCdevkit/VOC2012/SegmentationClassRaw',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'list_folder',
    os.path.join(work_dir, 'test_data', 'ImageSets', 'Segmentation'),
    'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string(
    'output_dir',
    os.path.join(work_dir, 'test_tfrecord'),
    'Path to save converted SSTable of TensorFlow examples.')

def _convert_dataset(dataset_split, sess, image_reader):
    dataset = os.path.basename(dataset_split)[:-4]
    filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
    num_images = 1
    num_per_shard = 1

    # image_reader = build.ImageReader(None, sess, 'jpeg', channels=3)

    output_filename = os.path.join(A_output_dir, '%s-00001-of-00001.tfrecord' % (dataset, ))

    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_idx = 0
        end_idx = 1
        i = 0
        print('\r>> Converting image 1')
        image_filename = os.path.join(A_image_folder, filenames[i] + '.' + A_image_format)
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)

        seg_data = image_data
        # convert to tf example
        example = build.image_seg_to_tfexample(image_data, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    print()

def write_record(sess, reader):
    dataset_splits = tf.gfile.Glob(os.path.join(A_list_folder, '*.txt'))
    for dataset_split in dataset_splits:
        _convert_dataset(dataset_split, sess, reader)
