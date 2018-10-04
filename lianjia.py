import os
import shutil
import argparse
import record
import tensorflow as tf


def convert2tfrecord(work_dir, filepath=''):
    # work_dir: '/assets/datasets'
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
    # image_folder: 'data/JPEGImages'
    # list_folder:  'data/ImageSets/Segmentation'
    # output_dir:   'tfrecord'
    command = 'python seg/deeplab/datasets/build_voc2012_data.py \
                --image_folder={} \
                --list_folder={} \
                --image_format="jpg"\
                --output_dir={}'.format(image_folder, list_folder, output_dir)
    os.system(command)


def segment(checkpoint_dir, work_dir):
    # checkpoint_dir: 'exp/train_on_train_set/train'
    # dataset_dir:    'tfrecord'
    dataset_dir = os.path.join(work_dir, 'tfrecord')
    command = 'python seg/deeplab/vis.py \
                --logtostderr \
                --vis_split="val" \
                --model_variant="xception_65" \
                --atrous_rates=6 \
                --atrous_rates=12 \
                --atrous_rates=18 \
                --output_stride=16 \
                --decoder_output_stride=4 \
                --vis_crop_size=1505 \
                --vis_crop_size=2049 \
                --dataset="homeplus" \
                --colormap_type=homeplus \
                --vis_batch_size=1 \
                --checkpoint_dir={} \
                --vis_logdir={} \
                --dataset_dir={}'.format(checkpoint_dir, work_dir, dataset_dir)
    os.system(command)


def segment2skels(work_dir):
    segment_folder = os.path.join(work_dir, 'segmentation_results')
    command = 'python seg2skel/seg2skels.py \
                --segment-folder={}'.format(segment_folder)
    os.system(command)
    input('before load skel')
    # command = 'python seg2skel/load_skels.py \
    #             --segment-folder={}'.format(segment_folder)
    # os.system(command)


def main(args):
    # convert2tfrecord(args.work_dir, args.filepath)
    segment(args.checkpoint_dir, args.work_dir)
    input('before seg2skel')
    segment2skels(args.work_dir)



if __name__ == '__main__':
    program_dir = '/home/chr/sgy/web/program/'
    parser = argparse.ArgumentParser(description='Segmentation and Skeleton')
    parser.add_argument('--work-dir', default=program_dir+'assets/datasets/', type=str)
    parser.add_argument('--checkpoint-dir', default=program_dir+'assets/datasets/exp/train_on_train_set/train/', type=str)
    parser.add_argument('--filepath', default='', type=str)
    args = parser.parse_args()

    # print(os.environ['PYTHONPATH'])

    main(args)
    # with tf.Session() as sess:
    #     record.write_record(None, sess)
    #     input('write finish')

[1505, 2049]
1
None
None
None
val
xception_65
[6, 12, 18]
16

[1505, 2049]
1
None
None
None
val
xception_65
[6, 12, 18]
16
