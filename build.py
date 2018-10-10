
import collections
import six
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_enum('image_format', 'png', ['jpg', 'jpeg', 'png'],
                         'Image format.')

# tf.app.flags.DEFINE_enum('label_format', 'png', ['png'],
#                          'Segmentation label format.')

tf.app.flags.DEFINE_enum('label_format', 'jpg', ['jpg', 'jpeg', 'png'],
                         'Segmentation label format.')

# A map from image format to expected data format.
_IMAGE_FORMAT_MAP = {
    'jpg': 'jpeg',
    'jpeg': 'jpeg',
    'png': 'png',
}

A_image_format = 'png'
A_label_format = 'jpg'

class ImageReader(object):

    def __init__(self, graph, sess, image_format='jpeg', channels=3):

        with graph.as_default():
        # with tf.get_default_graph():
        # graph = tf.get_default_graph()
        # self._decode_data = tf.placeholder(dtype=tf.string)
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._session = tf.Session(graph=graph)
            self._decode = tf.image.decode_jpeg(self._decode_data, channels=channels)


    def read_image_dims(self, image_data):
        image = self.decode_image(image_data)
        return image.shape[:2]

    def decode_image(self, image_data):
        image = self._session.run(self._decode, feed_dict={self._decode_data: image_data})
        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError('The image channels not supported.')

        return image


def _int64_list_feature(values):
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
    def norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def image_seg_to_tfexample(image_data, filename, height, width, seg_data):
    return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_list_feature(image_data),
      'image/filename': _bytes_list_feature(filename),
      'image/format': _bytes_list_feature(
          _IMAGE_FORMAT_MAP[A_image_format]),
      'image/height': _int64_list_feature(height),
      'image/width': _int64_list_feature(width),
      'image/channels': _int64_list_feature(3),
      'image/segmentation/class/encoded': (
          _bytes_list_feature(seg_data)),
      'image/segmentation/class/format': _bytes_list_feature(
          A_label_format),
    }))
