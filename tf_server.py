import os
import shutil
import argparse
import math
import time
import glob
import numpy as np

import tensorflow as tf
import record
import visualization
import build

from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator, save_annotation
from deeplab import common, model
# from seg2skel import seg2skel as seg_extract
# from seg2skel.utils import image as image_utils
# from seg2skel.utils import text as text_utils
# from seg2skel.utils import match_tpl as mtpl_utils
# from seg2skel.utils import skel as skel_utils
import skel_extract
import mycommon

slim = tf.contrib.slim

program_dir = '/home/chr/sgy/web/program'
work_dir = os.path.join(program_dir, 'assets', 'datasets')

A_vis_logdir = work_dir
A_checkpoint_dir = os.path.join(work_dir, 'exp', 'train_on_train_set', 'train/')

A_master = ''

reader_graph = None
reader = None
vis_graph = None
vis_session = None
supervisor = None
prepare_graph = None

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
A_colormap_type = 'homeplus'
A_eval_scales = [1.0]

_IMAGE_FORMAT = '{}_image'
_PREDICTION_FORMAT = '{}_prediction'
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw_segmentation_results'

def get_reader():
    global reader
    global reader_graph
    if reader is None:
        reader_graph = tf.Graph()
        reader = build.ImageReader(reader_graph, None, 'jpeg', 3)
    return reader

def get_supervisor():
    global supervisor
    global vis_graph
    if supervisor is None:
        vis_graph = tf.Graph()
        with vis_graph.as_default():
            tf.train.get_or_create_global_step()
            saver = tf.train.Saver(slim.get_variables_to_restore())
            supervisor = tf.train.Supervisor(init_op=tf.global_variables_initializer(),
                                             summary_op=None,
                                             summary_writer=None,
                                             global_step=None,
                                             saver=saver)
            with supervisor.managed_session(A_master, start_standard_services=False) as sess:
                supervisor.start_queue_runners(sess)
                supervisor.saver.restore(sess, tf.train.latest_checkpoint(A_checkpoint_dir))

    return supervisor

def get_session():
    global vis_graph
    global vis_session
    if vis_session is None:
        vis_graph = tf.Graph()
        with vis_graph.as_default():
            tf.train.get_or_create_global_step()
            vis_session = tf.Session(graph=vis_graph)
            saver = tf.train.Saver(slim.get_variables_to_restore())
            saver.restore(vis_session, tf.train.latest_checkpoint(A_checkpoint_dir))

    return vis_session

def get_prepare_graph():
    global prepare_graph
    if prepare_graph is None:
        prepare_graph = tf.Graph()
    return prepare_graph

def movefile(filepath):
    pascal_root = os.path.join(work_dir, 'data')
    output_dir = os.path.join(work_dir, 'tfrecord')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_folder = os.path.join(pascal_root, 'JPEGImages')
    list_folder = os.path.join(pascal_root, 'ImageSets/Segmentation')
    if filepath != '':
        fn = os.path.basename(filepath)
        local_fn = os.path.join(image_folder, fn)
        local_fn_list = os.path.join(list_folder, 'val.txt')
        if not os.path.exists(local_fn):
            shutil.copyfile(filepath, local_fn)
        with open(local_fn_list, 'w') as of:
            print(fn[:fn.rfind('.')])
            of.write(fn[:fn.rfind('.')])

def do_prepare():
    dataset = segmentation_dataset.get_dataset(A_dataset, A_vis_split, dataset_dir=A_dataset_dir)

    samples = input_generator.get(dataset,
                                  A_vis_crop_size,
                                  A_vis_batch_size,
                                  min_resize_value=A_min_resize_value,
                                  max_resize_value=A_max_resize_value,
                                  resize_factor=A_resize_factor,
                                  dataset_split=A_vis_split,
                                  is_training=False,
                                  model_variant=A_model_variant)

    model_options = mycommon.ModelOptions(
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

def my_process_batch(sess, original_images, semantic_predictions, image_names,
                   image_heights, image_widths, save_dir):
    (original_images,
     semantic_predictions,
     image_names,
     image_heights,
     image_widths) = sess.run([original_images, semantic_predictions,
                               image_names, image_heights, image_widths])

    # num_image = semantic_predictions.shape[0]
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

def do_process_batch(sess, samples, predictions):
    save_dir = os.path.join(work_dir, _SEMANTIC_PREDICTION_SAVE_FOLDER)

    my_process_batch(sess,
                   samples[common.ORIGINAL_IMAGE],
                   predictions,
                   samples[common.IMAGE_NAME],
                   samples[common.HEIGHT],
                   samples[common.WIDTH],
                   save_dir)

def process_one(filepath):
    global reader_graph
    global vis_graph

    movefile(filepath)
    with reader_graph.as_default():
        record.write_record(None, get_reader())


    # with get_prepare_graph().as_default():
    #     samples, predictions = do_prepare()
    #
    # with vis_graph.as_default():
    #     tf.train.get_or_create_global_step()
    #     sv = get_supervisor()
    #     with sv.managed_session(A_master, start_standard_services=False) as sess:
    #         do_process_batch(sess, samples, predictions)
    #         skel_extract.extract()
    #         skel_extract.load()

    with vis_graph.as_default():
        with get_session().as_default():
            samples, predictions = do_prepare()
            tf.train.get_or_create_global_step()
            do_process_batch(get_session(), samples, predictions)


def all_proc():
    get_reader()
    # get_supervisor()
    get_session()
    process_one('/home/chr/sgy/a.jpg')
    process_one('/home/chr/sgy/b.jpg')


def serve(filepath):
    global reader_graph
    movefile(filepath)

    # tf.reset_default_graph()

    get_reader()

    with reader_graph.as_default():
        record.write_record(None, get_reader())

    # input('record finish')


    vis_graph = tf.Graph()
    vis_graph.as_default()
    # with vis_graph.as_default():
        # -------------------------------------------

    samples, predictions = do_prepare()

    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(slim.get_variables_to_restore())
    # supervisor = tf.train.Supervisor(graph=vis_graph,
    supervisor = tf.train.Supervisor(init_op=tf.global_variables_initializer(),
                                     summary_op=None,
                                     summary_writer=None,
                                     global_step=None,
                                     saver=saver)

    with supervisor.managed_session(A_master, start_standard_services=False) as sess:

        supervisor.start_queue_runners(sess)
        my_checkpoint = tf.train.latest_checkpoint(A_checkpoint_dir)
        supervisor.saver.restore(sess, my_checkpoint)
        print(my_checkpoint)

        do_process_batch(sess, samples, predictions)

        skel_extract.extract()
        skel_extract.load()

    # graph.finalize()
    vis_graph.finalize()

# def do_skel():
#     segment_folder = os.path.join(work_dir, 'segmentation_results')
#     assets_dir = os.path.join(program_dir, 'seg2skel')
#     min_size = 500
#
#     photos = glob.glob(os.path.join(segment_folder, '*_image.png'))
#     text = seg_extract.Text(assets_dir)
#     skel = seg_extract.Skeleton()
#
#     for fpath in photos:
#         # example: fpath == 'segmentation_results/a_image.png'
#         print('process ', fpath)
#         ori_img = image_utils.imread(fpath)
#         if ori_img.shape[0] < min_size and ori_img.shape[1] < min_size:
#             continue
#
#         # seg_img == 'a_prediction.png'
#         seg_img = image_utils.imread(fpath[:-9] + 'prediction.png')
#
#         # output _text.json, according to origin image and (already generated) segmentation image
#         # with Timer('extract text'):
#         # ofpath == 'a_image_text_json'
#         ofpath = fpath[:-4] + '_text.json'
#         if not os.path.isfile(ofpath):
#             scale = text.extract(ori_img, seg_img, ofpath[:-5])
#         else:
#             scale = text.loadjson(ofpath)
#             # print('[[[[[[[[[[[[[')
#             # print(scale)
#             # print(']]]]]]]]]]]]]')
#             # input()
#         # with Timer('extract skels'):
#         ofpath = fpath[:-4] + '_skels.json'
#         skel.extract(ori_img, seg_img, ofpath, scale)
#         if not os.path.isfile(ofpath):
#             skel.extract(ori_img, seg_img, ofpath, scale)


if __name__ == '__main__':
    # print(A_checkpoint_dir)
    # serve('/home/chr/sgy/a.jpg')
    # serve_simple('/home/chr/sgy/a.jpg')
    # vis_origin('')
    all_proc()