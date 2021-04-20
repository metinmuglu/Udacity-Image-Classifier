# PROGRAMMER: Metin Muğlu
# DATE CREATED:12.20.2021

from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.utils.data
from torchvision import datasets, models, transforms
import argparse
import json
import time as  time

parser = argparse.ArgumentParser (description = "we can do it Parser of training script")
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier. Default value is 2048', type = int)
parser.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', type = str)
parser.add_argument ('--epochs', help = 'you can see output secreen of the Number of epochs', type = int)
parser.add_argument ('--GPU', help = "Cuda maybe Option to use GPU", type = str)
parser.add_argument ('--arch', help = 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used', type = str)
parser.add_argument ('--lrn', help = 'Learning rate, default value 0.001', type = float)
parser.add_argument ('data_dir', help = 'Provide data directory. must argument', type = str)

args = parser.parse_args ()
data_dir = args.data_dir
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
train_dir = data_dir + '/train'


#nessary fr input gPU condition
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'
    
# I always use to cpu
device = 'cpu'

#fixed condition
if data_dir: 
    train_transforms = transforms.Compose ([transforms.RandomRotation (30),
                                                transforms.RandomResizedCrop (224),
                                                transforms.RandomHorizontalFlip (),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])
    valid_transforms = transforms.Compose ([transforms.Resize (255),
                                                transforms.CenterCrop (224),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])
    test_transforms = transforms.Compose ([transforms.Resize (255),
                                                transforms.CenterCrop (224),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])   
my_train_image_datasets = datasets.ImageFolder (train_dir, transform = train_transforms)
my_valid_image_datasets = datasets.ImageFolder (valid_dir, transform = valid_transforms)
my_test_image_datasets = datasets.ImageFolder (test_dir, transform = test_transforms)

train_loader = torch.utils.data.DataLoader(my_train_image_datasets, batch_size = 64, shuffle = True)
valid_loader = torch.utils.data.DataLoader(my_valid_image_datasets, batch_size = 64)
test_loader = torch.utils.data.DataLoader(my_test_image_datasets, batch_size = 64)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model (arch, hidden_units):
    if arch == 'vgg19':
        model = models.vgg19 (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
            classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1024)),
                          ('drop', nn.Dropout(p=0.5)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1024, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    elif arch == 'vgg13':
        model = models.vgg13 (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: 
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (25088, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    else: 
        arch = 'alexnet'
        model = models.alexnet (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (9216, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
    model.classifier = classifier
    return model, arch

def validation(model, valid_loader, criterion):
    model.to (device)
    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

model, arch = load_model (args.arch, args.hidden_units)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)

model.to (device) 
epochs = 8
steps = 0
cuda = torch.cuda.is_available()
print_every = 40

if cuda:
    model.cuda()
else:
    model.cpu()

running_loss = 0
accuracy = 0

#model is train date
start = time.time()
print('Training started')

for e in range (epochs):
    for ii, (inputs, labels) in enumerate (train_loader):
        steps = steps + 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad () 

        outputs = model.forward (inputs) 
        loss = criterion (outputs, labels) 
        loss.backward ()
        optimizer.step () 
        running_loss = running_loss + loss.item () 

        if steps % print_every == 0:
            model.eval () 
            with torch.no_grad():
                valid_loss, accuracy = validation(model, valid_loader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "\n              Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                  "Accuracy: {:.3f}%".format(accuracy/len(valid_loader)*100))

            running_loss = 0
            model.train()

# we can see total model train time
time_elapsed = time.time() - start
print("\n Total time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))

model.to ('cpu') 
model.class_to_idx = my_train_image_datasets.class_to_idx 

checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': arch,
              'mapping':    model.class_to_idx
             }
#save my model :name 
if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')

print('Model Name:checkpoint.pth saved.')
