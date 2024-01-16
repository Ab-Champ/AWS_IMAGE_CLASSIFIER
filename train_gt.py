import argparse
from re import A
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict


class Training:
    @staticmethod
    def initialize(data_dir):
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
       
        train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        valid_data_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        test_data_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        
        image_datasets = {}
        image_datasets["train_data"] = datasets.ImageFolder(train_dir, transform = train_data_transforms)
        image_datasets["valid_data"] = datasets.ImageFolder(valid_dir, transform = valid_data_transforms)
        image_datasets["test_data"] = datasets.ImageFolder(test_dir, transform = test_data_transforms)

        train_dataloader = torch.utils.data.DataLoader(image_datasets["train_data"], batch_size = 64, shuffle =True)
        valid_dataloader = torch.utils.data.DataLoader(image_datasets["valid_data"], batch_size = 32)
        test_dataloader = torch.utils.data.DataLoader(image_datasets["test_data"], batch_size = 32)
        
        print(f"Data loaded from {data_dir} directory.")
        return image_datasets, train_dataloader, valid_dataloader, test_dataloader
    
    @staticmethod
    def create_model(arch,h_u):
        if arch.lower() == "vgg13":
            model_dn = models.vgg13(pretrained=True)
        else:
            model_dn = models.densenet121(pretrained=True)
        
        for param in model_dn.parameters():
            param.requires_grad = False 
        
        if arch.lower() =='vgg13':
            classifier = nn.Sequential(OrderedDict([
                                ('dropout1', nn.Dropout(0.1)),
                                ('fc1', nn.Linear(25088,h_u)),
                                ('relu1', nn.ReLU()),
                                ('dropout2', nn.Linear(h_u, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
        else:
            classifier = nn.Sequential(OrderedDict([
                                ('dropout1', nn.Dropout(0.1)),
                                ('fc1', nn.Linear(1024,h_u)),
                                ('relu1', nn.ReLU()),
                                ('dropout2', nn.Linear(h_u, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
        model_dn.classifier = classifier
        print(f"Model built from {arch} and {hidden_units} hidden units.")
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model_dn.classifier.parameters(), lr=0.001)
        return model_dn, optimizer, criterion 
    
    @staticmethod
    def validation(device, model, valid_loader, criterion):  
        model.to(device)
        valid_loss = 0
        accuracy = 0
        for inputs, labels in valid_loader:
        
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model.forward(inputs)
            valid_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        
        return valid_loss, accuracy 
 
    @staticmethod
    def train_model(model_dn, trainloader, validloader, learning_rate, epochs, gpu, optimizer, criterion,):
        device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
        epochs = epochs
        print_every = 10
        batch = 0
    
        model_dn.to(device)
    
        for e in range (epochs): 
            running_loss = 0
            running_accuracy=0
            print("___Starting Epoch {}/{}___".format(e+1,epochs))
        
            for i, (inputs, labels) in enumerate (trainloader):
                batch += 1
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
            
                optimizer.zero_grad () 
            
                # Forward and backward passes
                outputs = model_dn(inputs) #calculating output
                loss = criterion (outputs, labels) #calculating loss
                loss.backward () 
                optimizer.step () 
            
                running_loss += loss.item () 
            
                ps = torch.exp(outputs)
                equality = (labels.data == ps.max(dim = 1)[1])
                running_accuracy += equality.type(torch.FloatTensor).mean()
            
                if batch % print_every == 0:
                    model_dn.eval ()
                    
                
                    with torch.no_grad():
                        valid_loss, accuracy = Training.validation(device, model_dn, validloader, criterion)
                    
                    print("Epoch: {}/{}..".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Training_Accuracy: {:.3f}".format((running_accuracy/print_every)*100),
                        "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                        "Validation Accuracy: {:.3f}%".format(accuracy/len(validloader)*100))
                
                    running_loss = 0
                    model_dn.train()
        return model_dn, optimizer, criterion
                
parser = argparse.ArgumentParser()

# Basic usage: python train.py data_directory
parser.add_argument('data_dir', action='store',
                    default = 'flowers',
                    help='Set directory to load training data')

# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
parser.add_argument('--save_dir', action='store',
                    default = '.',
                    dest='save_dir',
                    help='Set directory to save checkpoints')

# Choose architecture: python train.py data_dir --arch "vgg13"
parser.add_argument('--arch', action='store',
                    default = 'densenet121',
                    dest='arch',
                    help='Choose architecture: e.g., "vgg13"')

# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20

parser.add_argument('--learning_rate', action='store',
                    default = 0.001,
                    dest='learning_rate',
                    help='Choose architecture learning rate')

parser.add_argument('--hidden_units', action='store',
                    default = 512,
                    dest='hidden_units',
                    help='Choose architecture hidden units')

parser.add_argument('--epochs', action='store',
                    default = 4,
                    dest='epochs',
                    help='Choose architecture number of epochs')

# Use GPU for training: python train.py data_dir --gpu
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for training, set a switch to true')
parser.add_argument('--optimizer', action='store',
                    default='adam',
                    dest='optimizer',
                    help='Choose optimizer ---> "adam"')

parser.add_argument('--criterion', action='store',
                    default='nllloss',
                    dest = 'criterion',
                    help='Choose criterion: ----> "nllloss"')

parse_results = parser.parse_args()
data_dir = parse_results.data_dir
save_dir = parse_results.save_dir
arch = parse_results.arch
learning_rate = float(parse_results.learning_rate)
hidden_units = int(parse_results.hidden_units)
epochs = int(parse_results.epochs)
gpu = parse_results.gpu
optimizer = parse_results.optimizer
criterion = parse_results.criterion

# Load and preprocess data
train_obj = Training()
image_datasets, train_loader, valid_loader, test_loader = train_obj.initialize(data_dir)

# Building and training the classifier
model_init, optimizer_init, criterion_init = train_obj.create_model(arch, hidden_units)
model, optimizer, criterion = train_obj.train_model(model_init, train_loader, valid_loader, learning_rate, epochs, gpu, optimizer_init, criterion_init)

#Save the checkpoint 
model.to ('cpu') 
model.class_to_idx = image_datasets['train_data'].class_to_idx 

#creating dictionary 
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'mapping':    model.class_to_idx
             }        

torch.save (checkpoint, 'classifier_checkpoint.pth')
if save_dir == ".":
    save_dir_name = "current folder"
else:
    save_dir_name = save_dir + " folder"

print(f'Checkpoint saved to {save_dir_name}.')
            
        
    

    