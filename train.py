"""
    This program will train a dataset using a pre-trained model, vgg16 or alexnet. The program will print out training loss, validation loss, and validation accuracy as the network trains.

"""

#Import modules
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image
import io
import pathlib
import argparse


#Command Line Arguments
def get_args():
    parser = argparse.ArgumentParser(description='Image Classifier')

    parser.add_argument('--data_dir', type=str,
                    help='image dataset') 
    
    parser.add_argument('--arch', action = 'store', type=str, default="vgg16",
                    help='enter arch model architecture e.g. vgg16, alexnet etc.')
    
    parser.add_argument('--input_size', type = int, default = 25088,
                    help='enter input size')
                        
    parser.add_argument('--hidden_units',type=list, default = [4096, 1024],
                    help='customise classifier network by providing list of hidden layers')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='provide learning rate')
    
    parser.add_argument('--dropout', action = 'store', type=float, default = 0.5,
                    help ='provide dropout for training the model')
    
    parser.add_argument('--epochs', type=int, default=5,
                    help='provide number of epochs to train model')
    
    parser.add_argument('--output', type=int, default=102,
                    help='enter output size')
    
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                    help='directory for saving the checkpoint file')
    
    parser.add_argument('--gpu', default=True,
                    help='use GPU or CPU to train model: True = GPU, False = CPU')
    
    args = parser.parse_args() 
                        
    return args



#Load datasets
def load_data():
                        
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Image Normalizations
    mean_norm = [0.485, 0.456, 0.406]
    std_norm = [0.229, 0.224, 0.225]

    #Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation((30)),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean_norm, std_norm)])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean_norm, std_norm)])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean_norm, std_norm)])

    #Load the datasets with ImageFolder
    train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_set = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_set = datasets.ImageFolder(test_dir, transform=test_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    
    return train_set, valid_set, test_set, trainloader, validloader, testloader


#for label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
    
#Build and train your network
def build_model(arch, hidden_units, output_layer, learning_rate, dropout, device):
#     model =  getattr(models,arch)(pretrained=True)
#     in_features = model.classifier[0].in_features

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    arch == 'vgg16'
    model = models.vgg16(pretrained=True)
    
    #freeze parameters
    for param in model.parameters():
        param.requires_grad = False     
    
    #define calssifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, hidden_units[0])),
                                        ('relu1', nn.ReLU()),
                                        ('drop1', nn.Dropout(p=dropout)),
                                        ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                                        ('relu2', nn.ReLU()),
                                        ('drop2', nn.Dropout(p=dropout)),
                                        ('fc3', nn.Linear(hidden_units[1], 102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                       ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);

    return model, criterion, optimizer

    
    
print('Training in process...')

def train_model(model, trainset, trainloader, validloader, testloader, optimizer, criterion, epochs, device):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.train()
    print_every = 30 
    steps = 0
    running_loss = 0

    train_losses, validation_losses = [], []

    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
        
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            #Validation loop
            if steps % print_every == 0:
                model.eval()
            
                validation_loss = 0
                accuracy = 0
            
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                
                        logps = model(images)
                        loss = criterion(logps, labels)
                        validation_loss += loss.item()

                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                train_losses.append(running_loss/len(trainloader))
                validation_losses.append(validation_loss/len(validloader))
            
                print(f"Epoch: {epoch+1}/{epochs}.."
                      f"Training Loss: {running_loss/print_every:.3f}.."
                      f"Validation Loss: {validation_loss/len(validloader):.3f}..",
                      f"Accuracy: {accuracy/len(validloader):.3f}")
            
            running_loss = 0
            model.train()
    
    model.class_to_idx = trainset.class_to_idx
    
    print("Training completed!")

    
    #Validation on testset
    print('Test in progress...')
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        model.eval()
    
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            _, predicted = torch.max(log_ps.data, 1)
        
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

  
    model.class_to_idx = trainset.class_to_idx
                                           
    print(f'Total number of test images: {test_total}',
          f'Test Accuracy: {(test_correct/test_total)*100}%')
    
    print('Test completed!')
    return model


#save checkpoint
def save_checkpoint(model, save_dir, hidden_units, optimizer, epochs, arch):
    checkpoint = {'arch': arch,
                  'output_size': 102,
                  'hidden_units': hidden_units,
                  'epoch': epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer.state_dict()
                 }

    torch.save(checkpoint, save_dir)

    print('Checkpoint saved!')
    
    return checkpoint
    
    

def main():            
    with open('cat_to_name.json', 'r') as f:
        cat_2_name = json.load(f)
    
    # get user arguments
    args = get_args()
    arch = args.arch
    data_dir = args.data_dir
       
    device = args.gpu
    input_size = args.input_size
    hidden_units = args.hidden_units
    dropout = args.dropout
    learning_rate = args.learning_rate
    epochs = args.epochs
    save_dir = args.save_dir
    
    #print arguments
    print(f'Data directory: {data_dir}'
          f'Architecture: {arch}'
          f'Input size: {input_size}'
          f'Hidden units: {hidden_units}'
          f'Dropout: {dropout}'
          f'Learning rate: {learning_rate}'
          f'Epochs: {epochs}'
          f'Dir saved in: {save_dir}'
         )
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu==True else 'cpu') 
    print("Device:", device)

 
    #call load_data function
    train_set, valid_set, test_set, trainloader, validloader, testloader = load_data()
 
    #call build_model function                 
    model, criterion, optimizer = build_model(args.arch, args.hidden_units, args.output, args.learning_rate, args.dropout, device)
    
    #call train_model function
    model = train_model(model, train_set, trainloader, validloader, testloader, optimizer, criterion, args.epochs, device)
    
    #call save_checkpoint function
    checkpoint = save_checkpoint(model, args.save_dir, args.hidden_units, optimizer, args.epochs, args.arch)
    
     
if __name__ == '__main__':
    main()      





    
    
    