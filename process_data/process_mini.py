"""Creates the mini-ImageNet dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import sys

import zipfile
import numpy as np
import scipy.misc
import io

# download miniImageNet dataset from https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing ,and put it in ./data folder

# Make train, validation and test splits deterministic from one run to another
np.random.seed(2021 + 4 + 20)

def get_class_label_dict(class_label_addr):
    lines = [x.strip() for x in open(class_label_addr, 'r').readlines()]
    cld={}
    for l in lines:
        tl = l.split(' ')
        if tl[0] not in cld.keys():
            cld[tl[0]] = tl[2].lower()
    return cld


def load_embedding_dict(emb_addr):
    fin = io.open(emb_addr, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word_vec[tokens[0]] = list(map(float, tokens[1:]))
    return word_vec

def get_embeddings_for_labels(all_classes, cld, emb_dict):
    label_list = []
    emb_list = []
    no_emb = 0
    print(all_classes)
    print(len(all_classes))
    for c in all_classes:
        label_list.append(cld[c])
    print(label_list)
    print(len(label_list))
    for v in label_list:
        # check the embeddings of labels
        #print(v)
        labels = v.split('_')
        tmpv = np.zeros(300)
        tmpl = []
        c = 0
        for l in labels:
            if l in emb_dict.keys():
                tmpv += emb_dict[l]
                tmpl.append(l)
                c += 1
        if len(labels) != 1:
            if c != len(labels):
                print(v, c, tmpl)
        if c != 0:
            emb_list.append(tmpv / c)
        else:
            emb_list.append(tmpv)
            no_emb += 1
            print("no embedding for " + v)
    print(no_emb)
    return emb_list


def main(data_dir, output_dir, emb_addr, class_label_addr):
    print("loading the embedding dictionary....")
    cld = get_class_label_dict(class_label_addr)
    emb_dict = load_embedding_dict(emb_addr)
    for split in ('val', 'test', 'train'):
        # List of selected image files for the current split
        file_paths = []
        with open('{}.csv'.format(split), 'r') as csv_file:
            # Read the CSV file for that split, and get all classes present in
            # that split.
            reader = csv.DictReader(csv_file, delimiter=',')
            file_paths, labels = zip(
                *((os.path.join('images', row['filename']), row['label'])
                  for row in reader))
            all_labels = sorted(list(set(labels)))
        print("getting word embeddings....")
        emb_list = get_embeddings_for_labels(all_labels, cld, emb_dict)
        print("saving word embeddings...")
        np.savez(
            os.path.join(output_dir, 'few-shot-wordemb-{}.npz'.format(split)),
            features=np.asarray(emb_list))

        archive = zipfile.ZipFile(os.path.join(data_dir, 'images.zip'), 'r')

        # Processing loop over examples
        features, targets = [], []
        for i, (file_path, label) in enumerate(zip(file_paths, labels)):
            # Write progress to stdout
            sys.stdout.write(
                '\r>> Processing {} image {}/{}'.format(
                    split, i + 1, len(file_paths)))
            sys.stdout.flush()

            # Load image in RGB mode to ensure image.ndim == 3
            file_path = archive.open(file_path)
            image = scipy.misc.imread(file_path, mode='RGB')
            # Infer class from filename.
            label = all_labels.index(label)

            # Central square crop of size equal to the image's smallest side.
            height, width, channels = image.shape
            crop_size = min(height, width)
            start_height = (height // 2) - (crop_size // 2)
            start_width = (width // 2) - (crop_size // 2)
            image = image[
                start_height: start_height + crop_size,
                start_width: start_width + crop_size, :]

            # Resize image to 84 x 84.
            image = scipy.misc.imresize(image, (84, 84), interp='bilinear')

            features.append(image)
            targets.append(label)

        sys.stdout.write('\n')
        sys.stdout.flush()

        # Save dataset to disk
        features = np.stack(features, axis=0)
        targets = np.stack(targets, axis=0)
        permutation = np.random.permutation(len(features))
        features = features[permutation]
        targets = targets[permutation]
        np.savez(
            os.path.join(output_dir, 'few-shot-{}.npz'.format(split)),
            features=features, targets=targets)

data_dir = './data'
output_dir = './data/mini'
class_label_addr = './data/mini_imagenet_class_label_dict3.txt'
emb_addr = './data/glove.840B.300d.txt'

main(data_dir, output_dir, emb_addr, class_label_addr)
