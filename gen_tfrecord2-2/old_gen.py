import os
import tensorflow as tf
import numpy as np
from PIL import Image



def create_record(file_list, name):
    writer = tf.python_io.TFRecordWriter(name)
    for f in file_list:
        #img = Image.open(f[0])
        #img_raw = img.tobytes()
        img_raw = tf.gfile.GFile(f[0], 'rb').read()
        ex = int(f[1])
        box = list(map(int, f[2:]))
        print(ex, box)
        example = tf.train.Example(
            features=tf.train.Features(feature={
                "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                "ex": tf.train.Feature(int64_list=tf.train.Int64List(value=[ex])),
                "box": tf.train.Feature(int64_list=tf.train.Int64List(value=box))
            }))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    image_size = 224
    num_catagories = 7
    train_file = np.loadtxt('/disk/AffectNet/ygyd_train.txt', dtype=object, delimiter='\t')[]
    test_file = np.loadtxt('/disk/AffectNet/ygyd_test.txt', dtype=object, delimiter='\t')
    create_record(train_file, 'train.tfrecord')
    create_record(test_file, 'test.tfrecord')

# with tf.Session() as sess:
#     for f in train_file:
#         img_raw_data = tf.gfile.FastGFile(f[0], 'rb').read()
#         bbox = f[2:]
#         expression = f[1]
#         image = tf.image.decode_jpeg(img_raw_data)
#         origin_img_shape = tf.shape(image)
#         ratio = tf.cast(image_size / tf.maximum(origin_img_shape[0], origin_img_shape[1]), tf.float32)
#         bbox = bbox * ratio
#         image = tf.image.resize_images(image, (image_size, image_size), preserve_aspect_ratio=True)
#         img_shape = tf.shape(image)
#         short_edge_side = tf.argmin([img_shape[0], img_shape[1]])
#         bias_dist = tf.cast((image_size - tf.minimum(img_shape[0], img_shape[1])) / 2, tf.float32)
#         ling = tf.reshape(tf.zeros((1,)), [])
#         bias_dist = tf.cond(tf.equal(short_edge_side, 0),
#                             lambda: tf.stack([ling, bias_dist, ling, bias_dist], 0),
#                             lambda: tf.stack([bias_dist, ling, bias_dist, ling], 0))
#         bbox += bias_dist
#         bbox = tf.scatter_nd([[1], [0], [3], [2]], bbox, [4])
#         image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
#         expression = tf.one_hot(expression, num_catagories, 1, 0)
#         expression = tf.cast(expression, tf.float32)
#         example = tf.train.Example(
#             features=tf.train.Features(feature={
#                 "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.eval()])),
#                 "ex": tf.train.Feature(float_list=tf.train.FloatList(value=[expression])),
#                 "box": tf.train.Feature(int64_list=tf.train.Int64List(value=[bbox]))
#             }))
#         writer.write(example.SerializeToString())
#     writer.close()
