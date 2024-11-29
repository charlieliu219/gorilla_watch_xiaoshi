import os
import numpy as np
import torch
import cv2
from timm import create_model
from torchvision import transforms
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm  # For progress bars
import torch
from timm import create_model
import timm
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images and extract class codes from file names
def load_images_and_labels(folder):
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(os.path.join(folder, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            label = filename.split("_")[0]
            labels.append(label)  
            filenames.append(filename)
    
    return images, labels

images, labels = load_images_and_labels("/workspaces/gorilla_watch/video_data/bristol/cropped_frames_filtered")
print(len(images))
print(labels)



