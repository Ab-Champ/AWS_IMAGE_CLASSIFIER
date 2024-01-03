import argparse
import numpy as np
import torch
from PIL import Image
import json
from torch import nn, optim
from torchvision import datasets, transforms, models
import seaborn as sns
import matplotlib.pyplot as plt

class Predict_class :
    @staticmethod
    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        image = Image.open (image) #loading image
        width, height = image.size #original size
        
        if width > height: 
            height = 256
            image.thumbnail ((50000, height), Image.ANTIALIAS)
        else: 
            width = 256
            image.thumbnail ((width,50000), Image.ANTIALIAS)
            
        
        width, height = image.size 
        crop = 224
        left_crop = (width - crop)/2 
        top_crop = (height - crop)/2
        right_crop = left_crop + 224 
        bottom_crop = top_crop + 224
        image = image.crop ((left_crop, top_crop, right_crop, bottom_crop))
        
        #preparing numpy array
        np_image = np.array (image)/255 #to make values from 0 to 1
        np_image -= np.array ([0.485, 0.456, 0.406]) 
        np_image /= np.array ([0.229, 0.224, 0.225])
        
        np_image= np_image.transpose ((2,0,1))
        return np_image
    @staticmethod
    def predict(image_path, model, topkl=5):
        
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        
        #Implement the code to predict the class from an image file
        image = Predict_class.process_image(image_path) #loading image and processing it using above defined function
        
        img  = torch.from_numpy (image).type (torch.FloatTensor)
    
        img = img.unsqueeze (dim = 0) 
         
        with torch.no_grad ():
            output = model.forward (img)
        output_prob = torch.exp (output) #converting into a probability
        
        probs, indices = output_prob.topk (topkl)
        probs = probs.numpy () #converting to numpy array
        indices = indices.numpy () 
        
        probs = probs.tolist () [0] #converting to list
        indices = indices.tolist () [0]
   
        mapping = {val: key for key, val in
                    model.class_to_idx.items()
                    }
    
        classes = [mapping [item] for item in indices]
        classes = np.array (classes) #converting to Numpy array 

        return probs, classes
    @staticmethod
    def load_model(file_path):
        checkpoint = torch.load(file_path)
        model = models.densenet121(pretrained=True)  
    
        # Update the classifier
        model.classifier = checkpoint['classifier']
    
        # Load the state_dict
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
        if 'mapping' in checkpoint:
            model.class_to_idx = checkpoint['mapping']
    
        return model
    
# Get the command line input
if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument('image_path', action='store',
                        default = 'flowers/test/1/image_06743.jpg',
                        help='Path to image, e.g., "flowers/test/1/image_06743.jpg"')
    
    parser.add_argument('checkpoint', action='store',
                        default='classifier_checkpoint.pth',
                        help='Filename of the saved checkpoint, e.g., "classifier_checkpoint.pth"')

    # Return top KK most likely classes
    parser.add_argument('--top_k', action='store',
                        default = 5,
                        dest='top_k',
                        help='Return top KK most likely classes, e.g., 5')

    # Use a mapping of categories to real names
    parser.add_argument('--category_names', action='store',
                        default = 'cat_to_name.json',
                        dest='category_names',
                        help='File name of the mapping of flower categories to real names, e.g., "cat_to_name.json"')

    # Use GPU for inference
    parser.add_argument('--gpu', action='store_true',
                        default=False,
                        dest='gpu',
                        help='Use GPU for inference, set a switch to true')

    parse_results = parser.parse_args()

    image_path = parse_results.image_path
    checkpoint = parse_results.checkpoint
    top_k = int(parse_results.top_k)
    category_names = parse_results.category_names
    gpu = parse_results.gpu

    # Label mapping
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    #loading model
    model = Predict_class.load_model(checkpoint)
    #Create an object of class predict
    pred_obj = Predict_class()
    # Image preprocessing
    np_image = pred_obj.process_image(image_path)

    # Predict class and probabilities
    print(f"Predicting top {top_k} most likely flower names from image {image_path}.")

    probs, classes = pred_obj.predict(image_path,model,top_k)
    classes_name = [cat_to_name[item] for item in classes]
    
    print("\nFlower name (probability): ")
    print("---------------------------------")
    for i in range(len(probs)):
        print(f"{classes_name[i]} ({round(probs[i], 3)})")
    print("")