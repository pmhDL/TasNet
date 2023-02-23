"""generate the tieredImageNet dataset."""
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
import pickle as pkl
import cv2
from tqdm import trange

# download tieredImageNet dataset from https://drive.google.com/file/d/1Letu5U_kAjQfqJjNPWS_rdjJ7Fd46LbX/view?usp=sharing , and unzip it to the ./data folder

# Make train, validation and test splits deterministic from one run to another
np.random.seed(2021 + 4 + 20)

def get_class_label_dict(class_label_addr):
    lines = [x.strip() for x in open(class_label_addr, 'r').readlines()]
    cld = {}
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

def get_embeddings_for_labels(all_classes, emb_dict):
    emb_list = []
    no_emb = 0
    print(all_classes)
    print(len(all_classes))
    for v in all_classes:
        # check the embeddings of labels
        #print(v)
        labels = v.split(', ')
        tmpv = np.zeros(300)
        tmpl = []
        c = 0
        lw=labels[0].split(' ')
        if labels[0]=='limpkin' or labels[0]=='otterhound' or labels[0]=='barracouta' or labels[0]=='lycaenid':
            lw = labels[1].strip().split(' ')
        for l in lw:
            if l in emb_dict.keys():
                tmpv += emb_dict[l]
                tmpl.append(l)
                c += 1
        if len(lw) != 1:
            if c != len(lw):
                print(v, c, tmpl)
        if c != 0:
            emb_list.append(tmpv / c)
        else:
            emb_list.append(np.random.rand(300)*2-1)
            no_emb += 1
            print("no embedding for " + v)
    print(no_emb)
    return emb_list


def main(data_dir, output_dir, emb_addr):
    print("loading the embedding dictionary....")
    emb_dict = load_embedding_dict(emb_addr)
    for split in ('val', 'test', 'train'):
        # List of selected image files for the current split
        with open(data_dir + '/' + split + '_images_png.pkl', 'rb') as f:
            raw_data = pkl.load(f, encoding='latin1')
        data = np.zeros([len(raw_data), 84, 84, 3], dtype=np.uint8)
        for ii in trange(len(raw_data)):
            item = raw_data[ii]
            im = cv2.imdecode(item, 1)
            # print(im)
            data[ii] = im
        f = open(data_dir + '/' + split + '_labels.pkl', 'rb')
        label_set = pkl.load(f, encoding='latin1')
        labels = label_set['label_specific']
        all_labels = label_set['label_specific_str']
        print("getting word embeddings....")
        emb_list = get_embeddings_for_labels(all_labels, emb_dict)
        print("saving word embeddings...")
        np.savez(
            os.path.join(output_dir, 'few-shot-wordemb-{}.npz'.format(split)),
            features=np.asarray(emb_list))
        # Processing loop over examples
        features, targets = [], []
        for i, (image, label) in enumerate(zip(data, labels)):
            # Write progress to stdout
            sys.stdout.write(
                '\r>> Processing {} image {}/{}'.format(
                    split, i + 1, label))
            # Infer class from filename.
            # Central square crop of size equal to the image's smallest side.
            height, width, channels = image.shape
            crop_size = min(height, width)
            start_height = (height // 2) - (crop_size // 2)
            start_width = (width // 2) - (crop_size // 2)
            image = image[
                start_height: start_height + crop_size,
                start_width: start_width + crop_size, :]

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

data_dir = './data/tiered-imagenet'
output_dir = './data/tiered'
emb_addr = './data/glove.840B.300d.txt'

main(data_dir, output_dir, emb_addr)

