import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
from collections import Counter
from tqdm import tqdm
from timm import create_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from wrappers_supervised import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_checkpoint_path = "/workspaces/gorilla_watch/video_data/gorillawatch/gorillatracker/models/should-be-the-best-model_vit_large_dinoV2.ckpt"
checkpoint_best = torch.load(best_checkpoint_path, map_location=device)

# Path to test folder
test_folder = "/workspaces/gorilla_watch/video_data/gorillawatch/gorillatracker/datasets/cxl_faces_squared_openset_kfold-5/test"
val_folder = "/workspaces/gorilla_watch/video_data/gorillawatch/gorillatracker/datasets/cxl_faces_squared_openset_kfold-5/fold-0"
bristol_folder = ""

# choose the dataset to evaluate
test_folder = test_folder


img_size = 224

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class GorillaDataset(Dataset):
    def __init__(self, folder, transform, threshold=3):
        self.folder = folder
        self.transform = transform
        self.threshold = threshold
        
        self.images = [] 
        self.labels = []
        self.videos = []

        self.filtered_images = []
        self.filtered_labels = []
        self.filtered_videos = []
        
        
        # the dataset of all images
        for filename in os.listdir(folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(folder, filename)
                self.images.append(image_path)
                label = filename.split("_")[0] # Extract label from filename
                self.labels.append(label)
                video = filename.split("_")[1] + "_" + filename.split("_")[2]
                self.videos.append(video)
        
        # Organize data by label & video
        data_by_label = {}
        for image, label, video in zip(self.images, self.labels, self.videos):
            if label not in data_by_label:
                data_by_label[label] = {"images": [], "videos": {}}
            data_by_label[label]["images"].append(image)
            
            if video not in data_by_label[label]["videos"]:
                data_by_label[label]["videos"][video] = []
            data_by_label[label]["videos"][video].append(image)


        # Filter classes that meet the threshold and video spread requirements
        self.valid_classes = {
            label
            for label, data in data_by_label.items()
            if sum(len(images) >= 3 for images in data["videos"].values()) >= 2  
        }

        # the dataset that is going to be classified by KNN5 CV
        for image, label, video in zip(self.images, self.labels, self.videos):
            if label in self.valid_classes:
                self.filtered_images.append(image)
                self.filtered_labels.append(label)
                self.filtered_videos.append(video)
        
        # checking the filtering
        print("Num images before filtering:", len(self.images))
        print("Num images after filtering:", len(self.filtered_images))
        print("Num classes before filtering:", len(data_by_label))
        print("Num classes after filtering:", len(self.valid_classes))
        
        # for label, data in data_by_label.items():
        #     if label not in self.valid_classes:
        #         continue
        #     print(f"Label: {label}")
        #     print(f"Total images: {len(data['images'])}")
        #     for video, images in data["videos"].items():
        #         print(f"  Video: {video}, Image Count: {len(images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # generate embeddings for all the images (and only classify the ones that are in the valid_classes)
        image_path = self.images[idx]
        label = self.labels[idx]
        video = self.videos[idx]    
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, label, video

# Create dataset and DataLoader
test_dataset = GorillaDataset(test_folder, transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
