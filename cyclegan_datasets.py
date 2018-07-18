"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'horse2zebra_train': 1000,
    'horse2zebra_test': 140,
    'photo2avatar': 7500,
    'photo2avatar_celeb': 600,
    'photo2avatar_test': 26,
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'horse2zebra_train': '*.jpg',
    'horse2zebra_test': '*.jpg',
    'photo2avatar': '*.jp*g',
    'photo2avatar_celeb': '*.jp*g',
    'photo2avatar_test': '*.jp*g',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'horse2zebra_train': './input/horse2zebra/horse2zebra_train.csv',
    'horse2zebra_test': './input/horse2zebra/horse2zebra_test.csv',
    'photo2avatar': './input/photo2avatar/photo2avatar.csv',
    'photo2avatar_celeb': './input/photo2avatar/photo2avatar_celeb.csv',
    'photo2avatar_test': './input/photo2avatar/photo2avatar_test.csv',
}
