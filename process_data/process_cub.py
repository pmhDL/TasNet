import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import scipy.misc
import random

# Download cub dataset from https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view, and unzip it to the data folder

data_path = './data/CUB_200_2011/images'
savedir = './data/cub/'
dataset_list = ['train', 'val', 'test']

#if not os.path.exists(savedir):
#    os.makedirs(savedir)

embpath = './data/CUB_200_2011/attributes/class_attribute_labels_continuous.txt'
emb = np.loadtxt(embpath)

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list, range(0, len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append([join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'train' in dataset:
            if (i%2 == 0):  #(i%4==0 and i%4==2)
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            if (i%4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'test' in dataset:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    features = []
    targets = []
    for k, file_path in enumerate(file_list):
        image = scipy.misc.imread(file_path, mode='RGB')
        height, width, channels = image.shape
        crop_size = min(height, width)
        start_height = (height // 2) - (crop_size // 2)
        start_width = (width // 2) - (crop_size // 2)
        image = image[
                start_height: start_height + crop_size,
                start_width: start_width + crop_size, :]
        # Resize image to 84 x 84.
        image = scipy.misc.imresize(image, (84, 84), interp='bilinear')
        label = label_list[k]
        features.append(image)
        targets.append(label)
    features = np.stack(features, axis=0)
    targets = np.stack(targets, axis=0)
    permutation = np.random.permutation(len(features))
    features = features[permutation]
    targets = targets[permutation]

    embv = emb[np.unique(targets), :]
    if 'train' in dataset:
        targets1 = (targets/2).astype(int)
    if 'val' in dataset:
        targets1 = ((targets-1)/4).astype(int)
    if 'test' in dataset:
        targets1 = ((targets-3)/4).astype(int)
        
    np.savez(os.path.join(savedir, 'few-shot-{}.npz'.format(dataset)), features=features, targets=targets1)
    np.savez(os.path.join(savedir, 'few-shot-wordemb-{}.npz'.format(dataset)), features=embv)

# embpath = './data/CUB_200_2011/attributes/class_attribute_labels_continuous.txt'
# emb = np.loadtxt(embpath)
# np.savez(os.path.join(savedir, 'few-shot-wordemb.npz'), features=emb)
