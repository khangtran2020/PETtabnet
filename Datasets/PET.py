from torch.utils.data import Dataset
import torch

class PETDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'target': self.one_hot(2, self.target[idx])
        }

    def get_targets(self):
        return list(self.target)

    @staticmethod
    def one_hot(size, target):
        tensor = torch.zeros(size, dtype=torch.float32)
        tensor[target] = 1.
        return tensor