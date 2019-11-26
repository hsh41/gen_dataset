import os
from math import ceil
import pandas as pd
import cv2
import tensorflow as tf
import pdb

first_time = False
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
image_size = 256
data_dir = '/data/FGVC/cub200-2011/CUB_200_2011'


def gen(df, parts, name, batch_size):
    batch = 0
    writer = []
    for i in range(parts):
        writer.append(tf.python_io.TFRecordWriter(name + '_' + str(i) + '.tfrecord'))
    iterator = fetch_data(df, batch_size)
    img, label = iterator.get_next()
    sess = tf.Session()
    while True:
        try:
            im, ex = sess.run([img, label])
        except tf.errors.OutOfRangeError:
            break
        sel = (batch * batch_size) // ceil(len(df) / parts)
        if batch % 500 == 0:
            print('batch ', batch, 'len ', len(ex), 'ex ', ex, 'sel ', sel)
        w = writer[sel]
        for i in range(len(ex)):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[im[i]])),
                        "ex": tf.train.Feature(int64_list=tf.train.Int64List(value=[ex[i]]))
                    }))
            w.write(example.SerializeToString())
        batch += 1
    for i in range(parts):
        writer[i].close()


def fetch_data(df, batch_size, num_epochs=1):
    def _features_parse_function(filename, bbox):
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # image = tf.slice(image, [bbox[1], bbox[0], 0], [bbox[3], bbox[2], 3])
        image = tf.image.resize_images(image, (image_size, image_size), preserve_aspect_ratio=True)
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        image = tf.image.encode_png(image)
        return image

    def _labels_parse_function(expression):
        return expression

    file_path = tf.convert_to_tensor(df.data_path.values, tf.string)
    dirs = tf.convert_to_tensor(os.path.join(data_dir, 'images/'), tf.string)
    file_path = tf.string_join([dirs, file_path])
    bbox = tf.convert_to_tensor(df[['x0', 'y0', 'w', 'l']].values, tf.int32)
    cls = tf.convert_to_tensor(df.cls.values, tf.int32)
    images = tf.data.Dataset.from_tensor_slices((file_path, bbox))
    labels = tf.data.Dataset.from_tensor_slices(cls)
    features = images.map(_features_parse_function, num_parallel_calls=6)
    labels = labels.map(_labels_parse_function, num_parallel_calls=6)
    dataset = tf.data.Dataset.zip((features, labels))
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    # dataset = dataset.shuffle(1000)
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def aug(df, times):
    return df.sample(frac=times, replace=True)


if __name__ == '__main__':
    data_df = pd.read_table(
        os.path.join(data_dir, 'images.txt'), sep=' ', header=None, names=['data_path'], index_col=0)
    bbox_df = pd.read_table(
        os.path.join(data_dir, 'bounding_boxes.txt'), sep=' ', header=None, names=['x0', 'y0', 'w', 'l'],
        index_col=0).astype(int)
    bbox_df[['w', 'l']] = bbox_df[['w', 'l']] - 1
    tt_df = pd.read_table(
        os.path.join(data_dir, 'train_test_split.txt'), sep=' ', header=None, names=['tt'], index_col=0)
    cls_df = pd.read_table(
        os.path.join(data_dir, 'image_class_labels.txt'), sep=' ', header=None, names=['cls'], index_col=0) - 1
    df = pd.concat([data_df, bbox_df, tt_df, cls_df], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    # num = 387
    # emit items that exceed the image
    if first_time:
        for num in range(len(df)):
            img = cv2.imread(os.path.join(data_dir, 'images', data_df['data_path'].iloc[num]))
            img = img[bbox_df['y0'].iloc[num]:bbox_df['l'].iloc[num] +
                      bbox_df['y0'].iloc[num], bbox_df['x0'].iloc[num]:bbox_df['x0'].iloc[num] + bbox_df['w'].iloc[num]]
            cv2.imwrite(os.path.join(data_dir, 'images', data_df['data_path'].iloc[num]), img)
    for n, g in df.groupby('tt'):
        # train
        if n == 1:
            pdb.set_trace()
            g = g.groupby('cls').apply(lambda x: aug(x, 5))
            g = g.sample(frac=1).reset_index(drop=True)
            print('train len', len(g))
            gen(g, 1, 'train', 1)
        # test
        elif n == 0:
            print('test len', len(g))
            gen(g, 1, 'test', 1)
