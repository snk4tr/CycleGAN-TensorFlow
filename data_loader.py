import os
import glob
import tensorflow as tf
import model


def get_batch(config: dict):
    dataset = _construct_dataset(config)
    dataset = _prepare(config, dataset)
    dataset = dataset.repeat(4)
    dataset = dataset.prefetch(buffer_size=config['buffer_size'])
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def _construct_dataset(config: dict) -> tf.data.Dataset:
    n_imgs = config['n_imgs']
    imgs_i, imgs_j = _get_file_names(config['path_domainA'], n_imgs), _get_file_names(config['path_domainB'], n_imgs)
    dataset = tf.data.Dataset.from_tensor_slices((imgs_i, imgs_j))
    if config['shuffle']:
        dataset = dataset.shuffle(buffer_size=config['n_imgs'], seed=42)
    dataset = dataset.map(_load)
    return dataset


def _get_file_names(path: str, n_imgs: int) -> tf.constant:
    return tf.constant(list(glob.glob(os.path.join(path, '*.*')))[:n_imgs])


def _load(img_i_path: str, img_j_path: str) -> (tf.Tensor, tf.Tensor):
    img_i_file, img_j_file = tf.read_file(img_i_path), tf.read_file(img_j_path)
    img_i_decoded = tf.image.decode_image(img_i_file, channels=3)
    img_j_decoded = tf.image.decode_image(img_j_file, channels=3)
    img_i_decoded.set_shape([None, None, 3]), img_j_decoded.set_shape([None, None, 3])
    return img_i_decoded, img_j_decoded


def _prepare(config: dict, dataset: tf.data.Dataset) -> tf.data.Dataset:
    dataset = dataset.map(lambda x, y: _do_resize(x, y, config['img_size']), num_parallel_calls=config['n_threads'])
    if config['augment']:
        dataset = dataset.map(_do_augment, num_parallel_calls=config['n_threads'])
    dataset = dataset.map(_normalize, num_parallel_calls=config['n_threads'])
    return dataset


def _do_resize(img_i: tf.Tensor, img_j: tf.Tensor, size: int, ) -> (tf.Tensor, tf.Tensor):
    return tf.image.resize_images(img_i, [size, size]), tf.image.resize_images(img_j, [size, size])


def _do_augment(img_i: tf.Tensor, img_j: tf.Tensor) -> (tf.constant, tf.constant):
    img_i = tf.image.random_flip_left_right(img_i)
    img_i = tf.random_crop(img_i, [model.IMG_HEIGHT, model.IMG_WIDTH, 3])
    return img_i, img_j


def _normalize(img_i: tf.Tensor, img_j: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    return tf.subtract(tf.div(img_i, 127), 1), tf.subtract(tf.div(img_j, 127), 1)
