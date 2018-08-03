"""Contains losses used for performing image-to-image domain adaptation."""
import tensorflow as tf


def cycle_consistency_loss(real_images, generated_images):
    """Compute the cycle consistency loss.

    The cycle consistency loss is defined as the sum of the L1 distances
    between the real images from each domain and their generated (fake)
    counterparts.

    This definition is derived from Equation 2 in:
        Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
        Networks.
        Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros.


    Args:
        real_images: A batch of images from domain X, a `Tensor` of shape
            [batch_size, height, width, channels].
        generated_images: A batch of generated images made to look like they
            came from domain X, a `Tensor` of shape
            [batch_size, height, width, channels].

    Returns:
        The cycle consistency loss.
    """
    return tf.reduce_mean(tf.abs(real_images - generated_images))


def lsgan_loss_generator(prob_fake_is_real):
    """Computes the LS-GAN loss as minimized by the generator.

    Rather than compute the negative loglikelihood, a least-squares loss is
    used to optimize the discriminators as per Equation 2 in:
        Least Squares Generative Adversarial Networks
        Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, and
        Stephen Paul Smolley.
        https://arxiv.org/pdf/1611.04076.pdf

    Args:
        prob_fake_is_real: The discriminator's estimate that generated images
            made to look like real images are real.

    Returns:
        The total LS-GAN loss.
    """
    return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
    """Computes the LS-GAN loss as minimized by the discriminator.

    Rather than compute the negative loglikelihood, a least-squares loss is
    used to optimize the discriminators as per Equation 2 in:
        Least Squares Generative Adversarial Networks
        Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, and
        Stephen Paul Smolley.
        https://arxiv.org/pdf/1611.04076.pdf

    Args:
        prob_real_is_real: The discriminator's estimate that images actually
            drawn from the real domain are in fact real.
        prob_fake_is_real: The discriminator's estimate that generated images
            made to look like real images are real.

    Returns:
        The total LS-GAN loss.
    """
    return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) +
            tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0))) * 0.5


def color_loss(input_a: tf.Tensor, input_b: tf.Tensor) -> tf.Tensor:
    hair_loss = _part_of_color_loss(input_a, input_b, 0.1, 0.35, 0.2, 0.3)
    shirt_loss = _part_of_color_loss(input_a, input_b, 0.85, 0.3, 0.15, 0.4)
    clr_loss = 2 * hair_loss + shirt_loss
    return clr_loss


def _part_of_color_loss(input_a, input_b, off_height, off_width, tar_height, tar_width):
    x_shape, y_shape = tf.shape(input_a)[1], tf.shape(input_a)[2]
    offset_height = tf.to_int32(tf.multiply(tf.to_float(x_shape), off_height))
    offset_width = tf.to_int32(tf.multiply(tf.to_float(y_shape), off_width))
    target_height = tf.to_int32(tf.multiply(tf.to_float(x_shape), tar_height))
    target_width = tf.to_int32(tf.multiply(tf.to_float(y_shape), tar_width))
    crop_a = tf.image.crop_to_bounding_box(input_a, offset_height, offset_width, target_height, target_width)
    crop_b = tf.image.crop_to_bounding_box(input_b, offset_height, offset_width, target_height, target_width)
    return tf.losses.mean_squared_error(crop_a, crop_b)
