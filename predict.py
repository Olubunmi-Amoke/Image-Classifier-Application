"""
    This program will predict flower name from an image, from the dataset provided, along with the probability of that name.
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


def get_args():
    
    parser = argparse.ArgumentParser(description='Image Classifier ')

    parser.add_argument('--image_path', action='store', type=str, 
                    default='flowers/test/2/image_05100.jpg', 
                    help='path to the image to be processed and visualized')
    
    parser.add_argument('--save_dir', type=str, action='store', default='checkpoint.pth',
                    help='directory to get the saved checkpoint from')
    
    parser.add_argument('--arch', type=str, default="vgg16",
                    help="input model architecture")
    
    parser.add_argument('--gpu', default=True,
                    help='use GPU or CPU to train model: True = GPU, False = CPU')
    
    parser.add_argument('--top_k', default=5, type=int, 
                    help="Number of Top probabilities")                
    
    parser.add_argument('--cat_to_name', action='store', default = 'cat_to_name.json',
                    help='Input path to image files')

    args = parser.parse_args() 
                        
    return args
                        
def load_checkpoint(path):
    """
       Loads a saved checkpoint.pth file
    """
    
    #checkpoint = torch.load(save_dir)
    checkpoint=torch.load(path, map_location=lambda storage, loc: storage)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
 
    model = getattr(models,arch)(pretrained=True)
    in_features = model.classifier[0].in_features
    
    for param in model.parameters():
        param.requires_grad=False
        
    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096)),
                                            ('relu1', nn.ReLU()),
                                            ('drop1', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(4096, 1024)),
                                            ('relu2', nn.ReLU()),
                                            ('drop2', nn.Dropout(p=0.5)),
                                            ('fc3', nn.Linear(1024, 102)),
                                            ('output', nn.LogSoftmax(dim=1))
                                           ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])


    return model

    
    
def process_image(image):
    """
    Function to load and pre-process test image
    
    Scales, crops, and normalizes a PIL image for a PyTorch model,returns a Numpy array
    """
                        
    # Load Image
    img = Image.open(image)#.convert("RGB")
    load_img = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ])
    
    image_tensor = load_img(img).float() 
    return image_tensor



def predict(image_path, model, top_k, cat_to_name, device):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    #Load and process the image 
    model.to(device) 
    
    image = process_image(image_path)
    
    image_tensor = image.to(device) 
    
    if device == 'cuda':
        image_tensor = image_tensor.unsqueeze_(0).cuda()
    else:
        image_tensor = image_tensor.unsqueeze_(0)
    
    with torch.no_grad():
        log_ps = model(image_tensor)
                
        ps = torch.exp(log_ps)
        probs, labels = ps.topk(5, dim=1)

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    probs = probs.cpu().numpy()[0]
    classes = []
    for label in labels.cpu().numpy()[0]:
        classes.append(idx_to_class[label]) 
        
    names = [cat_to_name[img_class] for img_class in classes]
    
    print(probs)
    print(classes)
    print(names)
    return probs, classes, names


def main():
    
    args = get_args()
    arch = args.arch
    image_path = args.image_path
    save_dir = args.save_dir
    device = args.gpu
    top_k = args.top_k
    cat_to_name = args.cat_to_name
    
    print(f'Architecture: {arch}'
          f'Flower image: {image_path}'
          f'Save Dir: {save_dir}'
          f'top_k: {top_k}'
          f'cat_to_name file: {cat_to_name}'
         )
    
    with open(args.cat_to_name, 'r') as f:
            cat_to_name = json.load(f)
     
    #call load_checkpoint function
    model = load_checkpoint(args.save_dir)
    
    #call process_image function
    image = process_image(args.image_path)
    
    #call predict function
    top_probabilities, top_classes, top_names = predict(args.image_path, model, args.top_k, cat_to_name, device)
    

if __name__ == "__main__":
    main()   