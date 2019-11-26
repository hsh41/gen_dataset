import os
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
image_size = 256


def create_record(data_dir, name, parts=1, batch_size=50, mode='train'):
    batch = 0
    writer = []
    for i in range(parts):
        writer.append(tf.python_io.TFRecordWriter(name + '_' + str(i) + '.tfrecord'))
    iterator = fetch_data(data_dir, batch_size, mode=mode)
    img, label = iterator.get_next()
    sess = tf.Session()
    while True:
        try:
            im, ex = sess.run([img, label])
        except tf.errors.OutOfRangeError:
            break
        sel = (batch * batch_size) // (210000 // parts)
        print('batch ', batch, 'len ', len(ex), 'sel ', sel)
        w = writer[sel]
        print('ex: ', ex)
        for i in range(len(ex)):
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[im[i]])),
                    "ex": tf.train.Feature(int64_list=tf.train.Int64List(value=[ex[i]]))
                }))
            w.write(example.SerializeToString())
        batch += 1
    for i in range(parts):
        writer[i].close()


def fetch_data(data_dir, batch_size, num_epochs=1, mode='train'):
    def _features_parse_function(filename, bbox):
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.slice(image, [bbox[0], bbox[1], 0], [bbox[2], bbox[3], 3])
        image = tf.image.resize_images(image, (image_size, image_size), preserve_aspect_ratio=True)
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        image = tf.image.encode_png(image)
        return image

    def _labels_parse_function(expression):
        return expression

    if mode == 'train':
        file_name = np.loadtxt(os.path.join(data_dir, './augmented_data2.txt'), dtype=object, delimiter=' ')
        np.random.shuffle(file_name)
    elif mode == 'test':
        file_name = np.loadtxt(os.path.join(data_dir, 'ygyd_test.txt'), dtype=object, delimiter=' ')
        np.random.shuffle(file_name)
    file_path = tf.convert_to_tensor(file_name[:, 0], tf.string)
    bbox = tf.convert_to_tensor(file_name[:, 1:5], tf.int32)
    expression = tf.convert_to_tensor(file_name[:, -1], tf.int32)
    images = tf.data.Dataset.from_tensor_slices((file_path, bbox))
    labels = tf.data.Dataset.from_tensor_slices(expression)
    features = images.map(_features_parse_function, num_parallel_calls=6)
    labels = labels.map(_labels_parse_function, num_parallel_calls=6)
    dataset = tf.data.Dataset.zip((features, labels))
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    # dataset = dataset.shuffle(100)
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()
    return iterator


if __name__ == '__main__':
    # data_dir = '/disk/AffectNet/'
    data_dir = ''
    create_record(data_dir, 'train', parts=1, batch_size=100, mode='train')
    print('test if fin')
    create_record(data_dir, 'test', parts=1, batch_size=100, mode='test')
