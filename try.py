import numpy as np
import tensorflow as tf
from glob import glob

inp_shape = (1, 2, 3)
folder = '/home/roman/Data/avatars/testA/*.jpg'

def create_graph():
    l = list(glob(folder))
    print(l)
    filename_queue = tf.train.string_input_producer(l[:2])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    image_decoded_A = tf.image.decode_jpeg(value)
    return image_decoded_A


def main():
    l = list(glob(folder))
    print(len(l))
    filename_queue = tf.train.string_input_producer(l)

    image_reader = tf.WholeFileReader()
    fname, image_file = image_reader.read(filename_queue)
    # img = tf.image.decode_jpeg(value)
    img = tf.image.decode_jpeg(image_file)

    res = set()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(len(l)):
            data = sess.run(img)
            data = sess.run(fname)
            res.add(data)
        coord.request_stop()
        coord.join(threads)
    print(len(res))

def main1():
    folder = '/home/roman/Dev/CycleGAN-leehomyc/input/photo2avatar/'
    filename_queue = tf.train.string_input_producer([folder + "photo2avatar_fei.csv", folder + "photo2avatar_celeb.csv"])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    # record_defaults = [[1], [1]]
    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]
    col1, col2 = tf.decode_csv(value, record_defaults=record_defaults)

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1200):
            # Retrieve a single instance:
            example, label = sess.run([col1, col2])
            print(example)

        coord.request_stop()
        coord.join(threads)

main()