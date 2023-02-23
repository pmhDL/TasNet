import pickle
import numpy as np
import matplotlib.pyplot as plt
# download FC100 from https://drive.google.com/file/d/1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1/view?usp=sharing
# download CIFAR-FS from https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing
# unzip the files to the ./data folder

path = './data'   #the path of saving data
FS = path + '/CIFAR-FS/'
FC = path + '/FC100/'
split = 'val'     # train  test  val
FSpath = FS + 'CIFAR_FS_' + split + '.pickle'
FC100path = FC + 'FC100_' + split + '.pickle'
f = open(FSpath, 'rb')
FS_data = pickle.load(f, encoding='latin1')
FS_Lb = np.array(FS_data['labels'])
FS_Im = FS_data['data']

f = open(FC100path, 'rb')
FC100_data = pickle.load(f, encoding='latin1')
FC100_Lb = np.array(FC100_data['labels'])
FC100_Im = FC100_data['data']

emb100 = np.load(path+'/cifar100-wordemb.npz')['features']

Lb_FS = np.unique(FS_Lb)
Lb_FC = np.unique(FC100_Lb)

embFS = emb100[Lb_FS, ]
embFC = emb100[Lb_FC, ]

dict_FS = {}
dict_FC = {}
for i in range(len(Lb_FS)):
    dict_FS[Lb_FS[i]]=i
for i in range(len(Lb_FC)):
    dict_FC[Lb_FC[i]]=i

FS_Lb1 = []
FC100_Lb1 = []
for k in FS_Lb:
    FS_Lb1.append(dict_FS[k])

for k in FC100_Lb:
    FC100_Lb1.append(dict_FC[k])

FS_Lb1 = np.array(FS_Lb1)
FC100_Lb1 = np.array(FC100_Lb1)

#random the data
id1 = np.random.permutation(len(FS_Lb1))
FS_Im1 = FS_Im
FS_Im1 = FS_Im1[id1]
FS_Lb1 = FS_Lb1[id1]

id2 = np.random.permutation(len(FC100_Lb1))
FC100_Im1 = FC100_Im
FC100_Im1 = FC100_Im1[id2]
FC100_Lb1 = FC100_Lb1[id2]

#save word embeddings of cifar-fs and fc100 data
np.savez('./data/cifar-fs/' + 'few-shot-wordemb-'+split+'.npz', features=embFS)
np.savez('./data/fc100/' + 'few-shot-wordemb-'+split+'.npz', features=embFC)
#save cifar-FS and fc100 data
np.savez('./data/cifar-fs/' + 'few-shot-' + split + '.npz', features=FS_Im1, targets=FS_Lb1)
np.savez('./data/fc100/' + 'few-shot-' + split + '.npz', features=FC100_Im1, targets=FC100_Lb1)