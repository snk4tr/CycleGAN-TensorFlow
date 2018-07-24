import json
import click
import os

from cyclegan import CycleGAN


@click.command()
@click.option('--to_train',
              type=click.INT,
              default=1,
              help='Whether it is train or false.')
@click.option('--log_dir',
              type=click.STRING,
              default=None,
              help='Where the data is logged to.')
@click.option('--config_filename',
              type=click.STRING,
              default='train',
              help='The name of the configuration file.')
@click.option('--skip',
              type=click.BOOL,
              default=False,
              help='Whether to add skip connection between input and output.')
@click.option('--epoch',
              type=click.STRING,
              default=None,
              help='What epochs to test.')  # TODO: to load
def main(to_train, log_dir, config_filename, skip, epoch):
    """
    :param to_train: Specify whether it is training or testing. 1: training; 2:
     resuming from latest checkpoint; 0: testing.
    :param log_dir: The root dir to save checkpoints and imgs. The actual dir
    is the root dir appended by the folder with the name timestamp.
    :param config_filename: The configuration file.
    :param skip: A boolean indicating whether to add skip connection between
    input and output.
    :param epoch: A description for epoch which you wish to test. It could be an int value or 
    a string containing range description in format "left, right" or "left, right, step".

    """
    os.makedirs(log_dir, exist_ok=True)
    with open(config_filename, 'r') as f:
        config = json.load(f)

    lambda_a = float(config.get('_LAMBDA_A', 10.0))
    lambda_b = float(config.get('_LAMBDA_B', 10.0))
    pool_size = int(config.get('pool_size', 50))

    to_restore = (to_train == 2)
    base_lr = float(config.get('base_lr', 2e-4))
    max_step = int(config.get('max_step', 200))
    network_version = str(config['network_version'])
    dataset_name = str(config['dataset_name'])

    cyclegan_model = CycleGAN(pool_size, lambda_a, lambda_b, log_dir,
                              to_restore, base_lr, max_step, network_version,
                              dataset_name, skip, epoch, config)

    if to_train:
        cyclegan_model.train()
        return
    cyclegan_model.test()


if __name__ == '__main__':
    main()
