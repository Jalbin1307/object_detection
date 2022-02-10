from matplotlib import patches
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

# Anotation file
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
    bboxs[:, [0,2]] /= w    # xmin, xmax
    bboxs[:, [1,3]] /= h    # ymin, ymax
except : pass
if bboxs.shape[0]:
    result.append({'image_id':image_id, 'image_size':image_size, 'bboxs':bboxs, 'labels':labels})

result = result[0]

w = 448
h = 448

im = np.array(Image.open(image_f).convert('RGB').resize((w,h)), dtype=np.uint8)

fig, ax = plt.subplots(1, figsize=(7,7))


bb = result['bboxs']
la = result['labels']

ax.imshow(im)

for b, l in zip(bb, la):
    rect = patches.Rectangle((b[0]*w, b[1]*h), (b[2]-b[0])*w, (b[3]-b[1])*h, linewidth=1, edgecolor='r',facecolor='none')

    ax.add_patch(rect)
    props = dict(boxstyle='round', facecolor='red', alpha=0.9)
    plt.text(b[0]*w, b[1]*h, classes[l], fontsize=10, color='white', bbox=props)

plt.axis('off')    
plt.show()



# plt.imshow(image)
# plt.show()

