import torch

import numpy as np

import os, xmltodict
import os.path as pth
from PIL import Image

import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)