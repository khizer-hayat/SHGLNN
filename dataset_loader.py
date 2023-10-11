import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

class CustomDatasetLoader:
    def __init__(self, name, save_dir):
        self.name = name
        self.save_dir = save_dir

    def load_dataset(self):
        dataset = TUDataset(root=self.save_dir, name=self.name)
        return dataset

    def get_dataloader(self, batch_size=32, shuffle=True):
        dataset = self.load_dataset()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
