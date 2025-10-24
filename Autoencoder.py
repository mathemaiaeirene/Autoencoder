
import os 
from torch.utils.data import Dataset
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np


class LabeledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        for cls_name in classes:
            cls_path = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_path):
                fpath = os.path.join(cls_path, fname)
                self.image_paths.append(fpath)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

    
class UnlabeledDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.paths = [os.path.join(self.path, file) for file in os.listdir(self.path)]
        self.transform = transform

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        return image, path
        

class Architecture(nn.Module):
    def __init__(self, latent_dims: int):
        super(Architecture, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),   
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                            
            nn.Flatten(),                               
            nn.Linear(in_features=128*16*16, out_features=latent_dims)          
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 128*16*16),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()
        )

    def forward(self, X):
        x = self.encoder(X)
        x = self.decoder(x)
        return x
    

class Autoencoder:
    def __init__(self, latent_dims: int, batch: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.batch = batch
        self.model = Architecture(latent_dims=latent_dims).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        
    def fit(self, path: str, epochs: int):
        '''
        Train a model in an specified number of epochs.

        Args:
            path (str): Path of images directory.
            epochs (int): Numer of epochs in which the model will be trained. 
            size (int): The size that the image will be resized for training.
        '''
        
        dataset = UnlabeledDataset(path=path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0

            for X, _ in dataloader:
                pred = self.model(X.to(self.device))
                loss = criterion(pred, X.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X.size(0)

            epoch_loss /= len(dataloader.dataset)
        
            print(f"Epoch: {epoch+1:3}/{epochs:<3}   |   Loss: {epoch_loss:8.5f}")
    
    def encode(self, path: str):
        dataset = UnlabeledDataset(path=path, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=False)

        encoded = []

        self.model.eval()
        with torch.no_grad():
            for X, _ in dataloader:
                pred = self.model.encoder(X.to(self.device))
                encoded.extend(pred.detach().cpu().tolist())

        return np.array(encoded)
