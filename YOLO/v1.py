import torch

import numpy as np

import os, xmltodict
import os.path as pth
from PIL import Image


import matplotlib.pyplot as plt



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

annot_f = './ano/000555.xml'
image_f = './img/000555.jpg'


classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 
           'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 
           'motorbike', 'train', 'bottle', 'chair', 'dining table', 
           'potted plant', 'sofa', 'tv/monitor' ]


f = open(annot_f)
info = xmltodict.parse(f.read())['annotation']
image_id = info['filename']
image_size = np.array(tuple(map(int, info['size'].values()))[:2], np.int16)
w, h = image_size
box_objects = info['object']
labels = []
bboxs = []
result = []

for obj in box_objects:
    try:
        labels.append(classes.index(obj['name'].lower()))
        bboxs.append(tuple(map(int, obj['bndbox'].values())))
    except: pass

bboxs = np.asarray(bboxs, dtype=np.float64)

try:
    bboxs[:, [0,2]] /= w
    bboxs[:, [1,3]] /= h
except : pass
if bboxs.shape[0]:
    result.append({'image_id':image_id, 'image_size':image_size, 'bboxs':bboxs, 'labels':labels})

print(result)


# image = Image.open(image_f)
# plt.imshow(image)
# plt.show()

