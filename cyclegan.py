import numpy as np
import os
import random
import tensorflow as tf
import cyclegan_datasets
import data_loader, losses, model

from scipy.misc import imsave
from tqdm import tqdm
from tensorflow.python import debug as tf_debug

slim = tf.contrib.slim


class CycleGAN:
    def __init__(self, pool_size, lambda_a,
                 lambda_b, output_root_dir, to_restore,
                 base_lr, max_step, network_version, skip, epoch, config):

        self._pool_size = pool_size
        self._lambda_a = lambda_a
        self._lambda_b = lambda_b
        self._output_dir = output_root_dir
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._to_restore = to_restore
        self._base_lr = base_lr
        self._max_step = max_step
        self._network_version = network_version
        self._checkpoint_dir = os.path.join(self._output_dir, 'checkpoints')
        self._skip = skip
        self._epoch = epoch

        self.fake_images_A = np.zeros(
            (self._pool_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH,
             model.IMG_CHANNELS)
        )
        self.fake_images_B = np.zeros(
            (self._pool_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH,
             model.IMG_CHANNELS)
        )
        self.fname = None
        self.config = config

    def model_setup(self):
        """
        This function sets up the model to train.

        self.input_A/self.input_B -> Set of training images.
        self.fake_A/self.fake_B -> Generated images by corresponding generator
        of input_A and input_B
        self.lr -> Learning rate variable
        self.cyc_A/ self.cyc_B -> Images generated after feeding
        self.fake_A/self.fake_B to corresponding generator.
        This is use to calculate cyclic loss
        """
        # self.inputs = data_loader.prepare(
        #     self.dataset_name, self.size, self.config,
        #     True, self._do_flipping)
        #
        # self.input_a = self.inputs['images_i']
        # self.input_b = self.inputs['images_j']

        self.input_a = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS
            ], name="input_B")

        self.fake_pool_A = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS
            ], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS
            ], name="fake_pool_B")

        self.global_step = tf.train.get_or_create_global_step()

        self.num_fake_inputs = 0

        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
        }

        outputs = model.get_outputs(
            inputs, network=self._network_version, skip=self._skip)

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']

    def compute_losses(self):
        """
        In this function we are defining the variables for loss calculations
        and training model.

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Various trainer for above loss functions
        *_summ -> Summary variables for above loss functions
        """
        cycle_consistency_loss_a = \
            self._lambda_a * losses.cycle_consistency_loss(
                real_images=self.input_a, generated_images=self.cycle_images_a,
            )
        cycle_consistency_loss_b = \
            self._lambda_b * losses.cycle_consistency_loss(
                real_images=self.input_b, generated_images=self.cycle_images_b,
            )

        # lsgan_loss_a = losses.lsgan_loss_generator(self.prob_fake_a_is_real) + just_brightness_loss
        lsgan_loss_a = losses.lsgan_loss_generator(self.prob_fake_a_is_real)
        lsgan_loss_b = losses.lsgan_loss_generator(self.prob_fake_b_is_real)

        g_loss_A = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
        g_loss_B = cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a

        d_loss_A = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real,
        )
        d_loss_B = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real,
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        # for var in self.model_vars:
        #     print(var.name)

        # Summary variables for tensorboard
        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

    def save_images(self, sess, epoch, save_dir):
        """
        Saves input and output images.

        :param sess: The session.
        :param epoch: Currnt epoch.
        """
        img_rel_path = 'imgs'
        images_dir = os.path.join(save_dir, img_rel_path)
        names = ['inputA_', 'inputB_', 'fakeA_', 'fakeB_', 'cycA_', 'cycB_']

        for name in names:
            subdir_path = os.path.join(images_dir, name)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)

        with open(os.path.join(save_dir, 'epoch_' + str(epoch) + '.html'), 'w') as v_html:
            # header
            header = '<table><tr><td width="256px">'
            header += '</td><td width="256px">'.join(names)
            header += '</td></tr></table>'
            v_html.write(header)

            for i in range(self.config['n_imgs_to_save']):
                inputs = sess.run(self.inputs)

                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([
                    self.fake_images_a,
                    self.fake_images_b,
                    self.cycle_images_a,
                    self.cycle_images_b
                ], feed_dict={
                    self.input_a: np.expand_dims(inputs[0], 0),
                    self.input_b: np.expand_dims(inputs[1], 0)
                })

                tensors = [inputs[0], inputs[1], fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]

                for name, tensor in zip(names, tensors):
                    image_name = name + str(epoch) + "_" + str(i) + ".jpg"
                    fpath = os.path.join(images_dir, name, image_name)
                    imsave(fpath, ((tensor[0] + 1) * 127.5).astype(np.uint8))
                    v_html.write("<img src=\"" + os.path.join(img_rel_path, name, image_name) + "\">")
                v_html.write("<br>")

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        """
        This function saves the generated image to corresponding
        pool of images.

        It keeps on feeling the pool till it is full and then randomly
        selects an already stored image and replace it with new one.
        """
        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def train(self):
        self.inputs = data_loader.get_batch(self.config)

        self.model_setup()
        self.compute_losses()

        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=21)

        max_images = self.config['n_imgs']
        tf_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
            device_count={'GPU': 1}
        )

        with tf.Session(config=tf_config) as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.run(init)

            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self._output_dir)
            os.makedirs(self._output_dir, exist_ok=True)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Training Loop
            start_step = sess.run(tf.train.get_global_step())
            print('Starting at epoch =', start_step)
            pr_bar = tqdm(range(start_step, self.config['n_epochs']),
                          bar_format='{desc}|{bar}|{percentage:3.0f}% ETA: {remaining}')
            for epoch in pr_bar:
                pr_bar.set_description('Epoch %d/%d' % (epoch, self.config['n_epochs']))
                if epoch % 10 == 0:
                    saver.save(sess, os.path.join(self._checkpoint_dir, "cyclegan"), global_step=epoch)
                    self.save_images(sess, epoch, self._output_dir)

                # Dealing with the learning rate as per the epoch number
                if epoch >= self.config['n_epochs']:
                    break
                elif epoch < 100:
                    curr_lr = self._base_lr
                else:
                    curr_lr = self._base_lr - \
                              self._base_lr * (epoch - 100) / 100

                for i in range(0, max_images):
                    inputs = sess.run(self.inputs)
                    in_a = np.expand_dims(inputs[0], 0)
                    in_b = np.expand_dims(inputs[1], 0)

                    # Optimizing the G_A network
                    _, fake_B_temp, summary_str = sess.run(
                        [self.g_A_trainer,
                         self.fake_images_b,
                         self.g_A_loss_summ],
                        feed_dict={
                            self.input_a:
                                in_a,
                            self.input_b:
                                in_b,
                            self.learning_rate: curr_lr
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    fake_B_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                    # Optimizing the D_B network
                    _, summary_str = sess.run(
                        [self.d_B_trainer, self.d_B_loss_summ],
                        feed_dict={
                            self.input_a: in_a,
                            self.input_b: in_b,
                            self.learning_rate: curr_lr,
                            self.fake_pool_B: fake_B_temp1

                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str = sess.run(
                        [self.g_B_trainer,
                         self.fake_images_a,
                         self.g_B_loss_summ],
                        feed_dict={
                            self.input_a: in_a,
                            self.input_b: in_b,
                            self.learning_rate: curr_lr
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    fake_A_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str = sess.run(
                        [self.d_A_trainer, self.d_A_loss_summ],
                        feed_dict={
                            self.input_a: in_a,
                            self.input_b: in_b,
                            self.learning_rate: curr_lr,
                            self.fake_pool_A: fake_A_temp1
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    writer.flush()
                    self.num_fake_inputs += 1

                sess.run(tf.assign(self.global_step, epoch + 1))

            epoch = sess.run(tf.train.get_global_step())
            saver.save(sess, os.path.join(self._checkpoint_dir, "cyclegan"), global_step=epoch)
            self.save_images(sess, epoch, self._output_dir)

            total_t = pr_bar.last_print_t - pr_bar.start_t
            ave_time_per_iter = total_t / pr_bar.total
            pr_bar.close()
            total_str = 'Total time: %10.2f h, average time per iteration: %10.2f sec'
            print(total_str % (total_t / 3600.0, ave_time_per_iter))

            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)

    def _test_single(self, saver, epoch=None):
        chkpt_fname = None
        if type(epoch) == int:
            chkpt_fname = os.path.join(self._checkpoint_dir, 'cyclegan-%d' % epoch)
            if not tf.train.checkpoint_exists(chkpt_fname):
                print('Epoch %d checkpoint was not found. Loading latest checkpoint instead.')
                chkpt_fname = None
        if chkpt_fname is None:
            chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
            epoch = None

        with tf.Session() as sess:
            saver.restore(sess, chkpt_fname)

            # restore epoch number for latest checkpoint only
            if epoch is None:
                epoch = sess.run(tf.train.get_global_step())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            self.save_images(sess, epoch, os.path.join(self._output_dir, 'epoch_%d' % epoch))

            coord.request_stop()
            coord.join(threads)

    def test(self):
        self.inputs = data_loader.get_batch(self.config)
        self.model_setup()
        saver = tf.train.Saver()

        print("---TESTING THE RESULTS---")
        if self._epoch is None:
            self._test_single(saver)
        elif self._epoch.isdigit():
            self._test_single(saver, epoch=int(self._epoch))
        elif ',' in self._epoch:
            range_str = list(map(int, str(self._epoch).split(',')))
            first_epoch, last_epoch = range_str[:2]
            step = range_str[2] if len(range_str) == 3 else 1
            self._output_dir = os.path.join(self._output_dir, 'test')
            for e in range(first_epoch, last_epoch, step):
                self._test_single(saver, e)
        else:
            raise AttributeError('Stop testing. Unexpected epoch description parameter.')
