import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image
'''
Implement the autoencoder in a simple way because I stuck in the problem.
Style embedding concatenation is used in the decoder. But how to ensure that 
the concatenated tensor dimensions are correct and match the input dimensions of the decoder 
is a great challenge for me and I debug for a long time. However, I couldn't solve this problem.
Therefore, I have to select a easier way to try the part 3.
'''
# Simple Encoder-Decoder with Style Embedding
class AutoencoderWithStyle(nn.Module):
    def __init__(self, style_dim=128):
        super(AutoencoderWithStyle, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 64x32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128x16x16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*16*16, 1024),  # Flatten to a 1024-d latent space
        )
        
        # Style Embedding Layer
        self.style_fc = nn.Linear(style_dim, 1024)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1024, 128*16*16),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # 64x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1),  # 3x64x64
            nn.Sigmoid(),  # Output between 0 and 1 for image pixels
        )
    
    def forward(self, x, style_embedding):
        # Pass image and style through encoder
        image_features = self.encoder(x)
        
        # Merge style embedding with image features
        style_features = self.style_fc(style_embedding)
        
        combined_features = image_features + style_features  # Simple addition of image and style features
        
        # Decode combined features to generate output image
        output = self.decoder(combined_features)
        return output

# Function to save before and after images
def save_images(images, outputs, save_path, batch_idx=0):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Define file paths for the images
    before_image_path = os.path.join(save_path, f"before_{batch_idx}.png")
    after_image_path = os.path.join(save_path, f"after_{batch_idx}.png")
    
    # Save the images
    save_image(images, before_image_path)
    save_image(outputs, after_image_path)
    
    print(f"Images saved to {save_path}")

# Define your dataset directory path
dataset_path = "/scratch/lt2326-2926-h24/wikiart/train" 

# Set up image transformations 
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),        # Convert images to tensor
])

# Load the dataset from the directory using ImageFolder
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize the autoencoder model
model = AutoencoderWithStyle(style_dim=128)

# Dummy style embedding (normally you'd use a pre-trained model for style embedding)
dummy_style_embedding = torch.randn(4, 128)  # Assuming batch_size=4, style_dim=128

# Testing the model
for images, _ in data_loader:
    # Here we assume a mismatch: we will use the same style embedding for all images
    outputs = model(images, dummy_style_embedding)
    
    # Save the before and after images to the specified directory
    save_images(images, outputs, save_path="./", batch_idx=0)
    break  # Just run one batch for testing
