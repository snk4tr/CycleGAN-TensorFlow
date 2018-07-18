import numpy as np
import tensorflow as tf
from glob import glob
from os import path, makedirs
from imageio import imsave, imread
from tqdm import tqdm

folder = '/home/roman/Data/avatars/testA/*.jpg'


def load_model(sess, checkpoint_dir, epoch=100):
    # load model
    saver = tf.train.import_meta_graph(path.join(checkpoint_dir, 'cyclegan-0.meta'))
    if saver is None:
        print('graph not found')
    chkpt_fname = path.join(checkpoint_dir, 'cyclegan-%d' % epoch)
    if chkpt_fname is None:
        print('checkpoint not found')
        return
    saver.restore(sess, chkpt_fname)

    # obtain inputs and outputs
    graph = tf.get_default_graph()

    inputA = graph.get_tensor_by_name("input_A:0")
    inputB = graph.get_tensor_by_name("input_B:0")
    fakeA = graph.get_tensor_by_name("Model/g_B/t1:0")
    fakeB = graph.get_tensor_by_name("Model/g_A/t1:0")
    cycA = graph.get_tensor_by_name("Model/g_B_1/t1:0")
    cycB = graph.get_tensor_by_name("Model/g_A_1/t1:0")
    fname = graph.get_tensor_by_name("ReaderReadV2:0")
    # self.fake_images_a,
    # self.fake_images_b,
    # self.cycle_images_a,
    # self.cycle_images_b
    return [inputA, inputB], [fname, fakeA, fakeB, cycA, cycB]


def preprocess(inp_img):
    image_size_before_crop, IMG_HEIGHT, IMG_WIDTH = 286, 256, 256
    # Preprocessing:
    out = tf.image.resize_images(inp_img, [image_size_before_crop, image_size_before_crop])

    out = tf.random_crop(out, [IMG_HEIGHT, IMG_WIDTH, 3])

    out = tf.subtract(tf.div(out, 127.5), 1)
    out = tf.expand_dims(out, 0)
    return out


def save_images_multithread(num, sess, epoch, save_dir, images_op, inputs, outputs):
    """
    Saves input and output images.

    :param sess: The session.
    :param epoch: Currnt epoch.
    """
    img_rel_path = 'imgs'
    images_dir = path.join(save_dir, img_rel_path)
    names = ['inputA_', 'inputB_', 'fakeA_', 'fakeB_', 'cycA_', 'cycB_']

    for name in names:
        subdir_path = path.join(images_dir, name)
        if not path.exists(subdir_path):
            makedirs(subdir_path)

    with open(path.join(save_dir, 'epoch_' + str(epoch) + '.html'), 'w') as v_html:
        # header
        header = '<table><tr><td width="256px">'
        header += '</td><td width="256px">'.join(names)
        header += '</td></tr></table>'
        v_html.write(header)

        # images
        res = set()
        for i in range(0, num):
            img_dict, fn = sess.run(images_op)
            print(fn)

            # print("Saving image {}/{}".format(i, self._num_imgs_to_save))
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = \
                sess.run(outputs[1:], feed_dict={inputs[0]: img_dict['images_i'], inputs[1]: img_dict['images_i']}) # feed_dict={inputs[0]: img, inputs[1]: img})
            res.add(fn)
            # tensors = [input_vals['images_i'], input_vals['images_j'], fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]
            tensors = [img_dict['images_i'], img_dict['images_i'], fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]

            for name, tensor in zip(names, tensors):
                image_name = name + str(epoch) + "_" + str(i) + ".jpg"
                fpath = path.join(images_dir, name, image_name)
                imsave(fpath, ((tensor[0] + 1) * 127.5).astype(np.uint8))
                v_html.write("<img src=\"" + path.join(img_rel_path, name, image_name) + "\">")
            v_html.write("<br>")
        print(len(res), num)


def main_multithread():
    l = sorted(glob(folder))
    print(len(l))

    filename_queue = tf.train.string_input_producer(l)

    image_reader = tf.WholeFileReader()
    fname, image_file = image_reader.read(filename_queue)
    img = tf.image.decode_jpeg(image_file)
    images_op = preprocess({'image_i': img, 'image_j': img})

    res = set()
    with tf.Session() as sess:
        inputs, outputs = load_model(sess)
        if outputs is None:
            return

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        save_dir = './test111'
        save_images_multithread(len(l), sess, 0, save_dir, [images_op, fname], inputs, outputs)

        # print(111)
        # coord.request_stop()
        # print(222)
        # coord.join(threads)
        print(333)
    print(len(res))


def save_images(sess, epoch, save_dir, inputs, outputs, img_cache):
    """
    Saves input and output images.

    :param sess: The session.
    :param epoch: Currnt epoch.
    """
    img_rel_path = 'imgs'
    images_dir = path.join(save_dir, img_rel_path)
    names = ['inputA_', 'inputB_', 'fakeA_', 'fakeB_', 'cycA_', 'cycB_']

    for name in names:
        subdir_path = path.join(images_dir, name)
        if not path.exists(subdir_path):
            makedirs(subdir_path)

    with open(path.join(save_dir, 'epoch_' + str(epoch) + '.html'), 'w') as v_html:
        # header
        header = '<table><tr><td width="256px">'
        header += '</td><td width="256px">'.join(names)
        header += '</td></tr></table>'
        v_html.write(header)

        # images
        for i, img in enumerate(img_cache):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = \
                sess.run(outputs[1:], feed_dict={inputs[0]: img, inputs[1]: img})
            tensors = [img, img, fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]

            for name, tensor in zip(names, tensors):
                image_name = name + str(epoch) + "_" + str(i) + ".jpg"
                fpath = path.join(images_dir, name, image_name)
                imsave(fpath, ((tensor[0] + 1) * 127.5).astype(np.uint8))
                v_html.write("<img src=\"" + path.join(img_rel_path, name, image_name) + "\">")
            v_html.write("<br>")


def main():
    file_list = sorted(glob(folder))
    inp_img_op = tf.placeholder(tf.float32, shape=[None, None, 3])
    prepro_op = preprocess(inp_img_op)
    experiment_dir = '/home/roman/Data/avatars/experiments/11/'
    checkpoint_dir = path.join(experiment_dir, 'checkpoints')

    with tf.Session() as sess:
        img_cache = []
        for i, fn in enumerate(file_list):
            img = imread(fn)
            img = sess.run(prepro_op, feed_dict={inp_img_op:img})
            img_cache.append(img)

        start_epoch, end_epoch, step = 110, 110, 10
        pr_bar = tqdm(range(start_epoch, end_epoch+1, step),
                      bar_format='{desc}|{bar}|{percentage:3.0f}% ETA: {remaining}')
        for epoch in pr_bar:
            pr_bar.set_description('Epoch %d/%d' % (epoch, end_epoch))
            inputs, outputs = load_model(sess, checkpoint_dir, epoch)
            if outputs is None:
                return
            save_dir = path.join(experiment_dir, 'test/epoch_%d' % epoch)
            save_images(sess, 0, save_dir, inputs, outputs, img_cache)

        total_t = pr_bar.last_print_t - pr_bar.start_t
        ave_time_per_iter = total_t / pr_bar.total
        pr_bar.close()
        total_str = 'Total time: %10.2f h, average time per iteration: %10.2f sec'
        print(total_str % (total_t / 3600.0, ave_time_per_iter))

main()
