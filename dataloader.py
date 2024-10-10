import numpy as np
from pathlib import Path
from scipy.io import loadmat
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, random_split

class DataLoader(Dataset):  # Inherit from torch.utils.data.Dataset
    def __init__(self, dataset):
        self.dataset = dataset
        self.imgs = []
        self.models = {}
        self.img2model = {}
        self.init_dataset()
        
    def init_dataset(self):

        list_file = Path(f'./dataset/list/{self.dataset}.txt')
        with open(list_file, 'r') as f:
            lines = f.readlines()
            for img_idx, file in enumerate(Path(f'./dataset/{self.dataset}').glob('*')):
                img = Image.open(file).resize((256, 256))
                img = np.array(img)
                self.imgs.append(img)

                idx = str(file).split('/')[-1].removesuffix('.png')
                idx = int(idx) - 1
                self.img2model[img_idx] = lines[idx].strip()
            

        for file in Path('./dataset/model').glob(f'*{self.dataset}*'):
            data = loadmat(str(file))
            file_name = str(file).split('/')[-1]
            file_name = file_name.removesuffix('.mat')
            self.models[file_name] = data['voxel']


    def __getitem__(self, idx):
        img = self.imgs[idx] 
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        model_name = self.img2model[idx]
        model = self.models[model_name] 
        model = torch.tensor(model, dtype=torch.float32)  
        
        return model, img
     
    def collect_fn(self, batch):
        models, imgs = zip(*batch)
        
        models = torch.stack(models) 
        imgs = torch.stack(imgs)
        
        return models, imgs 

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    # Load the dataset
    dataset = DataLoader('chair')  # Assume the dataset is 'chair'

    # Define train/validation split ratio (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into train and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders for train and validation sets
    train_loader = TorchDataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=dataset.collect_fn)
    val_loader = TorchDataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=dataset.collect_fn)

    # Example of iterating over the train_loader
    for xi, yi in train_loader:
        print("Training 3D Models (xi) batch shape:", xi.shape)
        print("Training 2D Images (yi) batch shape:", yi.shape)
        break

    # Example of iterating over the val_loader
    for xi, yi in val_loader:
        print("Validation 3D Models (xi) batch shape:", xi.shape)
        print("Validation 2D Images (yi) batch shape:", yi.shape)
        break
