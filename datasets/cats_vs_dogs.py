###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Cats and Dogs Datasets
"""
import errno
import os
import shutil
import sys

import torch
import torchvision
from torchvision import transforms

from PIL import Image

import ai8x

import numpy as np

torch.manual_seed(0)
class bayer_filter:
    def __call__(self, img):
        img = np.array(img)
        out =  np.zeros(shape=[img.shape[0],img.shape[1],3],dtype=img.dtype)
        for j in range(1,img.shape[1],2):
            for i in range(0,img.shape[0],2):
                out[i,j,2] = img[i, j, 2]
                out[i,j,1] = 0
                out[i,j,0] = 0
        for j in range(0,img.shape[1],2):
            for i in range(0,img.shape[0],2):
                out[i,j,1] = img[i, j, 1]
                out[i,j,2] = 0
                out[i,j,0] = 0
        for j in range(1,img.shape[1],2):
            for i in range(1,img.shape[0],2):
                out[i,j,1] = img[i, j, 1]
                out[i,j,2] = 0
                out[i,j,0] = 0
        for j in range(0,img.shape[1],2):
            for i in range(1,img.shape[0],2):
                out[i,j,0] = img[i, j, 0]
                out[i,j,1] = 0
                out[i,j,2] = 0
        return out 

class bayer_filter_2:
    def __call__(self, img):
        img = np.array(img)
        out =  np.zeros(shape=[img.shape[0],img.shape[1],1],dtype=img.dtype)
        for j in range(1,img.shape[1],2):
            for i in range(0,img.shape[0],2):
                out[i,j] = img[i, j, 2]
        for j in range(0,img.shape[1],2):
            for i in range(0,img.shape[0],2):
                out[i,j] = img[i, j, 1]
        for j in range(1,img.shape[1],2):
            for i in range(1,img.shape[0],2):
                out[i,j] = img[i, j, 1]
        for j in range(0,img.shape[1],2):
            for i in range(1,img.shape[0],2):
                out[i,j] = img[i, j, 0]
        return out
        
        
class fold_data():
    def __call__(self, img):
        img_folded = None
        fold_ratio=2
        for i in range(fold_ratio):
            for j in range(fold_ratio):
                img_subsample = img[i::fold_ratio, j::fold_ratio, :]
                if img_folded is not None:
                    img_folded = np.concatenate((img_folded, img_subsample), axis=2)
                else:
                    img_folded = img_subsample
                
        return img_folded
        
def augment_affine_jitter_blur(orig_img):
    """
    Augment with multiple transformations
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
        transforms.CenterCrop((180, 180)),
        transforms.ColorJitter(brightness=.7),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        transforms.RandomHorizontalFlip(),
        ])
    return train_transform(orig_img)


def augment_blur(orig_img):
    """
    Augment with center crop and bluring
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((220, 220)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))
        ])
    return train_transform(orig_img)


def catsdogs_get_datasets(data, load_train=True, load_test=True, aug=2, bayer=False, fold=True):
    """
    Load Cats & Dogs dataset
    """
    (data_dir, args) = data
    path = data_dir
    dataset_path = os.path.join(path, "cats_vs_dogs")
    is_dir = os.path.isdir(dataset_path)
    if not is_dir:
        print("******************************************")
        print("Please follow the instructions below:")
        print("Download the dataset to the \'data\' folder by visiting this link: "
              "\'https://www.kaggle.com/datasets/salader/dogs-vs-cats\'")
        print("If you do not have a Kaggle account, sign up first.")
        print("Unzip the downloaded file and find \'test\' and \'train\' folders "
              "and copy them into \'data/cats_vs_dogs\'. ")
        print("Make sure that images are in the following directory structure:")
        print("  \'data/cats_vs_dogs/train/cats\'")
        print("  \'data/cats_vs_dogs/train/dogs\'")
        print("  \'data/cats_vs_dogs/test/cats\'")
        print("  \'data/cats_vs_dogs/test/dogs\'")
        print("Re-run the script. The script will create an \'augmented\' folder ")
        print("with all the original and augmented images. Remove this folder if you want "
              "to change the augmentation and to recreate the dataset.")
        print("******************************************")
        sys.exit("Dataset not found!")
    else:
        processed_dataset_path = os.path.join(dataset_path, "augmented")

        if os.path.isdir(processed_dataset_path):
            print("augmented folder exits. Remove if you want to regenerate")

        train_path = os.path.join(dataset_path, "train")
        test_path = os.path.join(dataset_path, "test")
        processed_train_path = os.path.join(processed_dataset_path, "train")
        processed_test_path = os.path.join(processed_dataset_path, "test")
        if not os.path.isdir(processed_dataset_path):
            os.makedirs(processed_dataset_path, exist_ok=True)
            os.makedirs(processed_test_path, exist_ok=True)
            os.makedirs(processed_train_path, exist_ok=True)

            # create label folders
            for d in os.listdir(test_path):
                mk = os.path.join(processed_test_path, d)
                try:
                    os.mkdir(mk)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        print(f'{mk} already exists!')
                    else:
                        raise
            for d in os.listdir(train_path):
                mk = os.path.join(processed_train_path, d)
                try:
                    os.mkdir(mk)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        print(f'{mk} already exists!')
                    else:
                        raise

            # copy test folder files
            test_cnt = 0
            for (dirpath, _, filenames) in os.walk(test_path):
                print(f'copying {dirpath} -> {processed_test_path}')
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        relsourcepath = os.path.relpath(dirpath, test_path)
                        destpath = os.path.join(processed_test_path, relsourcepath)

                        destfile = os.path.join(destpath, filename)
                        shutil.copyfile(os.path.join(dirpath, filename), destfile)
                        test_cnt += 1

            # copy and augment train folder files
            train_cnt = 0
            for (dirpath, _, filenames) in os.walk(train_path):
                print(f'copying and augmenting {dirpath} -> {processed_train_path}')
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        relsourcepath = os.path.relpath(dirpath, train_path)
                        destpath = os.path.join(processed_train_path, relsourcepath)
                        srcfile = os.path.join(dirpath, filename)
                        destfile = os.path.join(destpath, filename)

                        # original file
                        shutil.copyfile(srcfile, destfile)
                        train_cnt += 1

                        orig_img = Image.open(srcfile)

                        # crop center & blur only
                        aug_img = augment_blur(orig_img)
                        augfile = destfile[:-4] + '_ab' + str(0) + '.jpg'
                        aug_img.save(augfile)
                        train_cnt += 1

                        # random jitter, affine, brightness & blur
                        for i in range(aug):
                            aug_img = augment_affine_jitter_blur(orig_img)
                            augfile = destfile[:-4] + '_aj' + str(i) + '.jpg'
                            aug_img.save(augfile)
                            train_cnt += 1
            print(f'Augmented dataset: {test_cnt} test, {train_cnt} train samples')

    # Loading and normalizing train dataset
    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=processed_train_path,
                                                         transform=train_transform)
    else:
        train_dataset = None

    # Loading and normalizing test dataset
    if load_test:
        if (bayer and fold):
            test_transform = transforms.Compose([
                transforms.Resize((128, 128)), bayer_filter_2(),fold_data(),
                transforms.ToTensor(),
                ai8x.normalize(args=args)
            ])
        elif (bayer):
            test_transform = transforms.Compose([
                transforms.Resize((128, 128)), bayer_filter(),
                transforms.ToTensor(),
                ai8x.normalize(args=args)
            ])
        else:
            test_transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                ai8x.normalize(args=args)
            ])

        test_dataset = torchvision.datasets.ImageFolder(root=processed_test_path,
                                                        transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'cats_vs_dogs',
        'input': (3, 128, 128),
        'output': ('cat', 'dog'),
        'loader': catsdogs_get_datasets,
    },
]
