import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
from wikiart import WikiArtDataset  # Reuse the dataset class
import json
import argparse

'''
For part 1 and part 2 
'''
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")

args = parser.parse_args()

config = json.load(open(args.config))
trainingdir = config["trainingdir"]
# Define the autoencoder model
class WikiArtAutoencoder(nn.Module):
    def __init__(self):
        super(WikiArtAutoencoder, self).__init__()

        # Encoder: a series of convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Pooling to reduce spatial dimensions

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Decoder: a series of transposed convolution layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output range [0,1]
        )

    def forward(self, x):
        # Encoding
        x = self.encoder(x)

        # Decoding
        x = self.decoder(x)
        return x

    def get_latent_repr(self, x):
        # Get the compressed representation (output of the encoder)
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # Flatten the output

'''
For part 1 to solve class imbalance problem
'''
# Solve class imbalance problem (weighted loss)
def get_class_weights(dataset):
    class_counts = np.zeros(len(dataset.classes))
    for _, label in dataset:
        class_counts[label] += 1
    total_count = len(dataset)
    class_weights = total_count / (len(dataset.classes) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32)

# Train the autoencoder model
def train_autoencoder(epochs=2, batch_size=32, learning_rate=0.001, device="cpu"):
    # Load the dataset
    traindataset = WikiArtDataset(trainingdir, device=device)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    # Initialize the autoencoder model
    model = WikiArtAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss function
    criterion = nn.MSELoss()

    # Train the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)  # Compute reconstruction error
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader)}")

    # Save the model
    torch.save(model.state_dict(), "autoencoder.pth")

    return model

def cluster_and_visualize(model, dataset, device="cpu", cluster_output_file="cluster_visualization.png", pca_output_file="pca_label_visualization.png"):
    model.eval()
    features = []
    labels = []
    
    # Get the compressed representations from the autoencoder
    for images, label in DataLoader(dataset, batch_size=1):  # Use DataLoader to ensure batch processing
        images = images.to(device)
        with torch.no_grad():
            latent_repr = model.get_latent_repr(images)
            features.append(latent_repr.cpu().numpy())  # Append the feature of each image
            labels.append(label.cpu().numpy())          # Append the label of each image
    
    # Convert labels and feature list to 2D arrays
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0).flatten()

    # PCA dimensionality reduction to 2D
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Use KMeans for clustering
    kmeans = KMeans(n_clusters=len(dataset.classes))
    kmeans.fit(reduced_features)

    # Figure 1: Clustering results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', s=50, alpha=0.7)
    plt.title("Clustering of Compressed Representations (Autoencoder)")
    plt.colorbar(label="Cluster ID")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(cluster_output_file)
    print(f"Cluster visualization saved to {cluster_output_file}")
    
    # Figure 2: PCA distribution based on true labels
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='tab10', marker='o', s=50, alpha=0.7)
    plt.title("PCA of Compressed Representations by True Label")
    plt.colorbar(label="True Label")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(pca_output_file)
    print(f"PCA visualization by true labels saved to {pca_output_file}")

# Execute training and visualization
device = config["device"]
model = train_autoencoder(device=device)
dataset = WikiArtDataset(trainingdir, device=device)
cluster_and_visualize(model, dataset, device)
