'''generate label word embeddings for cifar100'''
import os
import numpy as np
import io
import pickle
# download cifar100 dataset from https://www.cs.toronto.edu/~kriz/cifar.html and unzip it to ./data folder
# download glove embedding zip file: glove.840B.300d.txt from https://nlp.stanford.edu/data/glove.840B.300d.zip and unzip it to ./data folder
path = './data/cifar-100-python/meta' #root of cifar100 data
f = open(path, 'rb')
cifar100_classes = pickle.load(f, encoding='latin1')
classname = cifar100_classes['fine_label_names']

def load_embedding_dict(emb_addr):
    fin = io.open(emb_addr, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    i = 0
    for line in fin:
        i=i+1
        print(i)
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
        labels = v
        tmpv = np.zeros(300)
        tmpl = []
        c = 0
        lw = labels.split('_')
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


emb_addr = './data/glove.840B.300d.txt' #glove word embedding
emb_dict = load_embedding_dict(emb_addr)
output_dir = './data'
print("getting word embeddings....")
emb_list = get_embeddings_for_labels(classname, emb_dict)
print("saving word embeddings...")
np.savez(os.path.join(output_dir, 'cifar100-wordemb.npz'), features=np.asarray(emb_list))
print('finish save')