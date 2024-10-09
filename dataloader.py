import numpy as np
from pathlib import Path
from scipy.io import loadmat
from PIL import Image
import torch
from torch.utils.data import DataLoader as TorchDataLoader

class DataLoader():
    def __init__(self, dataset):
        self.dataset = dataset
        self.imgs = []
        self.models = {}
        self.img2model = {}
        self.init_dataset()
        
    def init_dataset(self):
        for file in Path(f'./dataset/{self.dataset}').glob('*'):
            img = Image.open(file).resize((256, 256))
            img = np.array(img)
            self.imgs.append(img)

            idx = str(file).split('/')[-1].removesuffix('.png')
            idx = int(idx)

            self.img2model[idx] = ''

        for file in Path('./dataset/model').glob(f'*{self.dataset}*'):
            data = loadmat(str(file))
            file_name = str(file).split('/')[-1]
            file_name = file_name.removesuffix('.mat')
            self.models[file_name] = data['voxel']

        list_file = Path(f'./dataset/list/{self.dataset}.txt')

        with open(list_file, 'r') as f:
            for idx, line in enumerate(f):
                self.img2model[idx] = line.strip()

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
    dataset = DataLoader('chair')  # 假设数据集为 'chair'
    dataloader = TorchDataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collect_fn)
    
    for xi, yi in dataloader:
        print("3D Models (xi) batch shape:", xi.shape)
        print("2D Images (yi) batch shape:", yi.shape)  
        break 


