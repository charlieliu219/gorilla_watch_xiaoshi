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
test_folder = "/workspaces/gorilla_watch/video_data/gorillawatch/gorillatracker/datasets/cxl_all_split_60-25-15/test"

img_size = 224

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Custom dataset
class GorillaDataset(Dataset):
    def __init__(self, folder, transform, threshold=3):
        self.folder = folder
        self.transform = transform
        self.images = []
        self.labels = []
        self.videos = []

        # Collect and filter data
        temp_images = []
        temp_labels = []
        temp_videos = []
        
        for filename in os.listdir(folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(folder, filename)
                temp_images.append(image_path)
                label = filename.split("_")[0] # Extract label from filename
                temp_labels.append(label)
                video = filename.split("_")[1] + filename.split("_")[2]
                temp_videos.append(video)

        # Count label occurrences and filter classes with enough samples
        label_counts = Counter(temp_labels)
        valid_classes = {label for label, count in label_counts.items() if count >= threshold}

        for image_path, label, video in zip(temp_images, temp_labels, temp_videos):
            if label in valid_classes:
                self.images.append(image_path)
                self.labels.append(label)
                self.videos.append(video)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
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


# Load model
def extract_clean_state_dict_for_wrapper(checkpoint, wrapper_key="model_wrapper.", model_key="model."):
    state_dict = checkpoint.get('state_dict', checkpoint)
    cleaned_state_dict = {k.replace(wrapper_key, ''): v for k, v in state_dict.items()}
    return cleaned_state_dict

model_wrapper = TimmWrapper(
    backbone_name="vit_large_patch14_dinov2.lvd142m",
    embedding_size=256,
    embedding_id="", # possible values: "linear", ""
    dropout_p=0.0,
    pool_mode="none",
    img_size=224
)

cleaned_state_dict_wrapper = extract_clean_state_dict_for_wrapper(checkpoint_best)
model_wrapper.load_state_dict(cleaned_state_dict_wrapper, strict=False)

model_wrapper.to(device)
model_wrapper.eval()

# Generate embeddings
def generate_embeddings(model, data_loader):
    all_embeddings = []
    all_labels = []
    all_videos = []
    with torch.no_grad():
        for images, labels, videos in tqdm(data_loader, desc="Generating Embeddings"):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels)
            all_videos.extend(videos)
    return torch.cat(all_embeddings), all_labels, all_videos

embeddings, labels, video_ids = generate_embeddings(model_wrapper, test_loader)

# KNN5 Cross-Video Metric

def KNN5_CV(embeddings, labels, video_ids, distance_metric="euclidean"):
    embeddings = embeddings.numpy()
    num_neighbors = 5  # Number of valid neighbors to consider for classification

    # Function to calculate distance based on the chosen metric
    def calculate_distance(embeddings, test_embedding, metric):
        if metric == "euclidean":
            # Compute Euclidean distance
            distances = np.linalg.norm(embeddings - test_embedding, axis=1)
        elif metric == "cosine":
            # Normalize embeddings to unit vectors for cosine similarity
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_test_embedding = test_embedding / np.linalg.norm(test_embedding)
            # Compute cosine similarity
            cosine_similarity = np.dot(normalized_embeddings, normalized_test_embedding)
            # Convert similarity to distance
            distances = 1 - cosine_similarity
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
        return distances

    vit_y_pred = []

    for idx, test_embedding in enumerate(embeddings):
        # Calculate distances using the chosen metric
        distances = calculate_distance(embeddings, test_embedding, distance_metric)

        # Get sorted indices of neighbors based on distance
        sorted_indices = np.argsort(distances)
        sorted_indices = sorted_indices[sorted_indices != idx]  # Exclude self

        valid_neighbors = []
        for neighbor_idx in sorted_indices:
            if video_ids[neighbor_idx] != video_ids[idx]:
                valid_neighbors.append(neighbor_idx)
                if len(valid_neighbors) == num_neighbors:
                    break

        if len(valid_neighbors) < num_neighbors:
            print(f"Warning: Less than {num_neighbors} valid neighbors for index {idx}.")
        
        # Get labels for the valid neighbors
        valid_neighbor_labels = [labels[i] for i in valid_neighbors]

        if valid_neighbor_labels:
            # Predict the label based on majority vote
            predicted_label = max(set(valid_neighbor_labels), key=valid_neighbor_labels.count)
        else:
            # If no neighbors remain, assign a default label (e.g., "Unknown")
            predicted_label = "Unknown"

        vit_y_pred.append(predicted_label)

    # Calculate accuracy and F1 score (excluding "Unknown" predictions)
    valid_indices = [i for i, pred in enumerate(vit_y_pred) if pred != "Unknown"]
    valid_y_pred = [vit_y_pred[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]

    vit_accuracy = accuracy_score(valid_labels, valid_y_pred)
    print(f"KNN5 ({distance_metric.capitalize()} Distance) Cross-Video Accuracy: {vit_accuracy:.4f}")

KNN5_CV(embeddings, labels, video_ids, distance_metric="euclidean")