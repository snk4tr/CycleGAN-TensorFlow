import glob
import tensorflow as tf
import model


def _do_resize(images: tf.constant, size: int) -> tf.constant:
    return tf.image.resize_images(images, [size, size])


def _do_augment(images: tf.constant) -> tf.constant:
    images = tf.image.random_flip_left_right(images)
    images = tf.random_crop(images, [model.IMG_HEIGHT, model.IMG_WIDTH, 3])
    return images


def _normalize(images: tf.constant) -> tf.constant:
    return tf.subtract(tf.div(images, 127.5), 1)


def _load(path: str) -> tf.constant:
    return tf.constant(list(glob.glob(path)))


def _prepare(path: str, size: int, augment: bool, shuffle: bool):
    images = _load(path)
    images = _do_resize(images, size)
    if augment:
        images = _do_augment(images)
    images = _normalize(images)

    # Batch
    if shuffle:
        inputs['images_i'], inputs['images_j'] = tf.train.shuffle_batch(
            [inputs['image_i'], inputs['image_j']], 1, 5000, 100)

    inputs['images_i'], inputs['images_j'] = tf.train.batch(
        [inputs['image_i'], inputs['image_j']], 1)

    return inputs


def get_data(config):
    return _prepare(config['path_domainA'], config['size'], config['augment'], config['shuffle']), \
           _prepare(config['path_domainB'], config['size'], config['augment'], config['shuffle'])


def asd(config):
    images_i, images_j = _load(config['path_domainA']), _load(config['path_domainB'])
    dataset = tf.data.Dataset.from_tensor_slices((images_i, images_j))