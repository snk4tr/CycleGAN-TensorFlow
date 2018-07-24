import os
import glob
import tensorflow as tf
import model


def get_batch(config: dict):
    dataset = _construct_dataset(config)
    dataset = _prepare(config, dataset)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def _construct_dataset(config: dict) -> tf.data.Dataset:
    imgs_i, imgs_j = _get_filenames(config['path_domainA']), _get_filenames(config['path_domainB'])
    dataset = tf.data.Dataset.from_tensor_slices((imgs_i, imgs_j))
    dataset = dataset.map(_load)
    return dataset


def _get_filenames(path: str) -> tf.constant:
    return tf.constant(list(glob.glob(os.path.join(path, '*.*')))[:7])


def _load(img_i_path: str, img_j_path: str) -> (tf.Tensor, tf.Tensor):
    img_i_file, img_j_file = tf.read_file(img_i_path), tf.read_file(img_j_path)
    img_i_decoded = tf.image.decode_image(img_i_file, channels=3)
    img_j_decoded = tf.image.decode_image(img_j_file, channels=3)
    img_i_decoded.set_shape([None, None, 3]), img_j_decoded.set_shape([None, None, 3])
    return img_i_decoded, img_j_decoded


def _prepare(config: dict, dataset: tf.data.Dataset) -> tf.data.Dataset:
    dataset = dataset.map(lambda x, y: _do_resize(x, y, config['img_size']), num_parallel_calls=4)
    if config['augment']:
        dataset = dataset.map(_do_augment, num_parallel_calls=4)
    dataset = dataset.map(_normalize, num_parallel_calls=4)
    if config['shuffle']:
        dataset = dataset.shuffle(buffer_size=7500, seed=42)
    return dataset


def _do_resize(img_i: tf.constant, img_j: tf.constant, size: int) -> (tf.constant, tf.constant):
    return tf.image.resize_images([img_i], [size, size]), img_j


def _do_augment(img_i: tf.constant, img_j: tf.constant) -> (tf.constant, tf.constant):
    img_i = tf.image.random_flip_left_right(img_i)
    img_i = tf.random_crop(img_i, [model.IMG_HEIGHT, model.IMG_WIDTH, 3])
    return img_i, img_j


def _normalize(img_i: tf.constant, img_j: tf.constant) -> tf.constant:
    return tf.subtract(tf.div(img_i, 127), 1), tf.subtract(tf.div(img_j, 127), 1)
