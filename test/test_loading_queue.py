import json
import tensorflow as tf
import os

from optparse import OptionParser
from glob import glob


def data_loader(path) -> (tf.Tensor, tf.Tensor):
    filename_queue = tf.train.string_input_producer(list(glob(os.path.join(path, "*.*"))))
    image_reader = tf.WholeFileReader()
    fname, image_file = image_reader.read(filename_queue)
    img = tf.image.decode_jpeg(image_file)
    return img, fname


def transformer(size: int, path: str) -> (tf.Tensor, tf.Tensor):
    img, fname = data_loader(path)
    inp = tf.image.resize_images(img, [size, size])
    return inp, fname


def get_num_imgs(path: str) -> int:
    return len(list(glob(os.path.join(path, '*.*'))))


def init_params() -> (int, list):
    with open('../configs/photo2avatar.json', 'r') as f:
        config = json.load(f)
    size = config.get('img_size', 286)
    paths = [config.get('path_domainA'), config.get('path_domainB')]
    return size, paths


def main():
    parser = OptionParser(usage='usage: %prog [options] <paths>')
    parser.add_option('-v', '--verbose', type=int, help='0 - print less, 1 - print more', default=0)
    options, args = parser.parse_args()

    size, paths = init_params()
    print('---WORKING EXAMPLE---')
    run_works(size, paths, options.verbose)
    print('---BROKEN EXAMPLE')
    run_broken(size, paths, options.verbose)


def run_works(size: int, paths: list, verbose: int):
    with tf.Session() as sess:
        for i, path in enumerate(paths):
            print("---PATH #%d: %s---" % (i + 1, path))
            data = transformer(size, path)
            nimgs = get_num_imgs(path)

            for j in range(nimgs):
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                res = sess.run(data)
                if verbose:
                    print(j, res[1], res[0].shape)
            coord.request_stop()
            coord.join(threads)


def run_broken(size: int, paths: list, verbose: int):
    with tf.Session() as sess:
        for i, path in enumerate(paths):
            print("---PATH #%d: %s---" % (i + 1, path))
            data = transformer(size, path)
            nimgs = get_num_imgs(path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for j in range(nimgs):
                res = sess.run(data)
                if verbose:
                    print(j, res[1], res[0].shape)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main()
