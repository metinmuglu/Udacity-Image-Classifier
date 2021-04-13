# PROGRAMMER: Metin Muğlu
# DATE CREATED:12.02.2021
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json
import seaborn as sns

parser = argparse.ArgumentParser (description = "Parser for prediction code")

parser.add_argument ('--image_dir', help = 'path to image. must argument', type = str)
parser.add_argument ('--load_dir', help = 'path to model. must argument', type = str)
parser.add_argument ('--category_names', help = 'JSON file name to be provided ----Mapping of categories to real names. your side Optional', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. your side Optional', type = int)
parser.add_argument ('--GPU', help = "Use GPU. your side Optional", type = str)

def loading_model (file_path):
    checkpoint = torch.load (file_path) 
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else: 
        model = models.vgg13 (pretrained = True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']

    for param in model.parameters():
        param.requires_grad = False 

    return model

def process_image(image):
    ''' Scales, crops, and normalizes
    '''
    im = Image.open (image) 
    width, height = im.size 

    if width > height:
        height = 256
        im.thumbnail ((50000, height), Image.ANTIALIAS)
    else:
        width = 256
        im.thumbnail ((width,50000), Image.ANTIALIAS)

    width, height = im.size 
    reduce = 224
    left = (width - reduce)/2
    top = (height - reduce)/2
    right = left + 224
    bottom = top + 224
    im = im.crop ((left, top, right, bottom))

    np_image = np.array (im)/255 
    np_image -= np.array ([0.485, 0.456, 0.406])
    np_image /= np.array ([0.229, 0.224, 0.225])

    np_image= np_image.transpose ((2,0,1))
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    

    image = np.array (image)
    image = image.transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

def predict(image_path, model, topkl, device):
    '''  image using a trained for model.
    '''
    image = process_image (image_path) 


    if device == 'cuda':
        im = torch.from_numpy (image).type (torch.cuda.FloatTensor)
    else:
        im = torch.from_numpy (image).type (torch.FloatTensor)

    im = im.unsqueeze (dim = 0) 



    model.to (device)
    im.to (device)

    with torch.no_grad ():
        output = model.forward (im)
    output_prob = torch.exp (output) 

    probs, indeces = output_prob.topk (topkl)
    probs = probs.cpu ()
    indeces = indeces.cpu ()
    probs = probs.numpy ()
    indeces = indeces.numpy ()

    probs = probs.tolist () [0]
    indeces = indeces.tolist () [0]

    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }

    classes = [mapping [item] for item in indeces]
    classes = np.array (classes) 

    return probs, classes

args = parser.parse_args ()
file_path = args.image_dir

#if condition nessary same fail input
if args.load_dir == 'Result' or args.load_dir is None:
    model_path = 'result'
else:
    model_path = args.load_dir

#if condition nessary same fail input
if args.image_dir == 'Flowers' or args.image_dir is None:
    file_path = 'flowers'
else:
    file_path = args.image_dir
    
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

model = loading_model (args.load_dir)

if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 1

#for condition I check all image in folder --> loop by for
klasor = file_path  #'flowers/valid/1'
for i in os.listdir(klasor):
    dosya = os.path.join(klasor,i)
    if os.path.isdir(dosya):
        print ('File => ', i)
    elif os.path.isfile(dosya):
        print ('Image => ', i)
        probs, classes = predict (klasor+'/'+i, model, nm_cl, device)
        class_names = [cat_to_name [item] for item in classes]
        for l in range (nm_cl):
            print("Number: {}/{}.. ".format(l+1, nm_cl),
                "Class name: {}.. ".format(class_names [l]),
                "Probability: {:.3f}..% ".format(probs [l]*100),
                )  
