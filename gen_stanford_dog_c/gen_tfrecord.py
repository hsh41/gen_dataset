import os
from math import ceil
from xml.dom.minidom import parse
import scipy.io as sio
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import pdb

image_size = 256
first_time = False
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
            im, logits = sess.run([img, label])
        except tf.errors.OutOfRangeError:
            break
        sel = (batch * batch_size) // ceil(len(df) / parts)
        if batch % 500 == 0:
            print('batch ', batch, 'len ', len(logits), 'logits ', logits, 'sel ', sel)
        w = writer[sel]
        for i in range(len(logits)):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[im[i]])),
                        "logits": tf.train.Feature(int64_list=tf.train.Int64List(value=[logits[i]]))
                    }))
            w.write(example.SerializeToString())
        batch += 1
    for i in range(parts):
        writer[i].close()


def fetch_data(df, batch_size, num_epochs=1):
    def _features_parse_function(filename):
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, (image_size, image_size), preserve_aspect_ratio=True)
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        image = tf.image.encode_png(image)
        return image

    def _labels_parse_function(expression):
        return expression

    file_path = tf.convert_to_tensor(df.file_list.values, tf.string)
    dirs = tf.convert_to_tensor(os.path.join(dataset_path, 'Images/'), tf.string)
    file_path = tf.string_join([dirs, file_path])
    cls = tf.convert_to_tensor(df.labels.values, tf.int32)
    images = tf.data.Dataset.from_tensor_slices(file_path)
    labels = tf.data.Dataset.from_tensor_slices(cls)
    features = images.map(_features_parse_function, num_parallel_calls=6)
    labels = labels.map(_labels_parse_function, num_parallel_calls=6)
    dataset = tf.data.Dataset.zip((features, labels))
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    # dataset = dataset.shuffle(1000)
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def parse_xml(annotation_path, df):
    dom_obj = parse(os.path.join(annotation_path, df['annotation_list']))
    elem_obj = dom_obj.documentElement
    width = elem_obj.getElementsByTagName("width")[0].firstChild.data
    height = elem_obj.getElementsByTagName("height")[0].firstChild.data
    depth = elem_obj.getElementsByTagName("depth")[0].firstChild.data
    xmin = elem_obj.getElementsByTagName("xmin")[0].firstChild.data
    ymin = elem_obj.getElementsByTagName("ymin")[0].firstChild.data
    xmax = elem_obj.getElementsByTagName("xmax")[0].firstChild.data
    ymax = elem_obj.getElementsByTagName("ymax")[0].firstChild.data
    return pd.Series({
        'width': width,
        'height': height,
        'depth': depth,
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax
    })


def gen_df(dataset_path, mat_name):
    data = sio.loadmat(os.path.join(dataset_path, mat_name))
    data.pop('__header__')
    data.pop('__version__')
    data.pop('__globals__')
    # reduce the data to 1 dim
    data['file_list'] = np.concatenate(data['file_list'])
    data['file_list'] = np.concatenate(data['file_list'])
    data['annotation_list'] = np.concatenate(data['annotation_list'])
    data['annotation_list'] = np.concatenate(data['annotation_list'])
    data['labels'] = np.concatenate(data['labels']).astype(int) - 1
    annotation_path = os.path.join(dataset_path, 'Annotation')
    df = pd.DataFrame(data)
    df[['width', 'height', 'depth', 'xmin', 'ymin', 'xmax', 'ymax']] = df.apply(
        lambda x: parse_xml(annotation_path, x), axis=1).astype(int)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def aug(df, times):
    return df.sample(frac=times, replace=True)


def select_df(df, label=[1, 2, 3, 4, 5, 6, 7, 8]):
    if df['labels'].iloc[0] in label:
        return df
    else:
        return None


if __name__ == '__main__':
    dataset_path = '/data/FGVC/stanford dogs'
    train_df = gen_df(dataset_path, 'train_list.mat')
    if first_time:
        for num in range(len(train_df)):
            image_path = os.path.join(dataset_path, 'Images', train_df['file_list'].iloc[num])
            img = cv2.imread(image_path)
            img = img[train_df['ymin'].iloc[num]:train_df['ymax'].iloc[num], train_df['xmin'].
                      iloc[num]:train_df['xmax'].iloc[num]]
            cv2.imwrite(image_path, img)

    test_df = gen_df(dataset_path, 'test_list.mat')
    if first_time:
        for num in range(len(test_df)):
            image_path = os.path.join(dataset_path, 'Images', test_df['file_list'].iloc[num])
            img = cv2.imread(image_path)
            img = img[test_df['ymin'].iloc[num]:test_df['ymax'].iloc[num], test_df['xmin'].iloc[num]:test_df['xmax'].
                      iloc[num]]
            cv2.imwrite(image_path, img)
    print('first time is fin')
    train_df = train_df.groupby('labels').apply(lambda x: aug(x, 5))
    # train_df = train_df.sample(frac=1, replace=True)
    # select 8 classes
    # train_df = train_df.set_index('labels').sort_index().reset_index()
    # test_df = test_df.set_index('labels').sort_index().reset_index()
    # train_df = train_df.groupby('labels').apply(select_df)
    # test_df = test_df.groupby('labels').apply(select_df)
    gen(train_df, 8, 'train', 1)
    gen(test_df, 1, 'test', 1)
