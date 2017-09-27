from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale

from dataset import DatasetFromFolder1



# def  data_dir(dest, upscale_factor):
#   output_dir = join(dest+)


#data processing

def input_transform(crop_size, upscale_factor):
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
    return Compose([
        Scale((crop_size // upscale_factor,crop_size // upscale_factor)),
        ToTensor(),
        ])

def target_transform(crop_size, upscale_factor):
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
    return Compose([
                Scale((crop_size,crop_size)),
                ToTensor(),
                ])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def get_test_set(dir1,dir2):
    test_dir = dir1
    label_dir = dir2

    return DatasetFromFolder2(test_dir,label_dir,
            input_transform=input1_transform(),
            target_transform=input1_transform())

def get_training_set(dir1,dir2):
    train_dir = dir1
    label_dir = dir2
    return DatasetFromFolder1(train_dir, label_dir,
            input_transform=input_transform(128,2),
            target_transform=target_transform(128,2))


