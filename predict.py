# PROGRAMMER: Metin Muğlu
# DATE CREATED:12.02.2021

from PIL import Image
import argparse
import json
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torchvision import models
import torch.utils.data

parser = argparse.ArgumentParser (description = "Let's edit the parser parameters for the prediction code.")

parser.add_argument ('--image_dir', help = 'path to image. must argument', type = str)
parser.add_argument ('--load_dir', help = 'path to model. must argument', type = str)
parser.add_argument ('--category_names', help = 'JSON file name to be provided ----Mapping of categories to real names. your side Optional', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. your side Optional', type = int)
parser.add_argument ('--GPU', help = "Use GPU. your side Optional", type = str)

def process_image(image):
    im = Image.open(image)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2,0,1)

# for example yor model checkpointPTH.pth
def load_checkpoint (filename):
    checkpointPTH = torch.load (filename)
    if checkpointPTH ['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else: 
        model = models.vgg13 (pretrained = True)
    
    model.class_to_idx = checkpointPTH ['mapping']
    model.classifier = checkpointPTH['classifier']
    model.load_state_dict(checkpointPTH['state_dict'])

    return model


def imshow(image, ax=None):
    """Imshow for Tensor."""
    image = np.array (image)
    image = image.transpose((1, 2, 0))
    if ax is None:
        fig, ax = plt.subplots()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])  
    
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

def predict(image_path, model, topkl, device):
    '''  
        image using a trained for model.
    '''
    image = process_image (image_path) 


    if device == 'cuda':
        im = torch.from_numpy (image).type (torch.cuda.FloatTensor)
        print("We go for cuda")
    else:
        im = torch.from_numpy (image).type (torch.FloatTensor)
        print("We go for CPU")

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
filename = args.image_dir

#if condition nessary same fail input
if args.load_dir == 'Result' or args.load_dir is None:
    model_path = 'result'
else:
    model_path = args.load_dir

#if condition nessary same fail input
if args.image_dir == 'Flowers' or args.image_dir is None:
    filename = 'flowers'
else:
    filename = args.image_dir
    
if args.GPU == 'GPU':
    device = 'cpu'
else:
    device = 'cuda'

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

model = load_checkpoint (args.load_dir)

if args.top_k:
    number_cl = args.top_k
else:
    number_cl = 1

#for condition I check all image in folder --> loop by for
klasor = filename  #'flowers/valid/1'
for i in os.listdir(klasor):
    dosya = os.path.join(klasor,i)
    if os.path.isdir(dosya):
        print ('File => ', i)
    elif os.path.isfile(dosya):
        print ('Image => ', i)
        probs, classes = predict (klasor+'/'+i, model, number_cl, device)
        names_class = [cat_to_name [item] for item in classes]
        for l in range (number_cl):
            print("Number: {}/{}.. ".format(l+1, number_cl),
                "Name to Class: {}.. ".format(names_class [l]),
                "Model Probability: {:.3f}..% ".format(probs [l]*100),
                )  
