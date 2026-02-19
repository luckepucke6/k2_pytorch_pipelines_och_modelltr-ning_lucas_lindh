import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MyDataset(Dataset):
    def __init__(self, train=True):
        # Hur bilderna konverteras till tensorer
        self.transform = transforms.ToTensor()

        # Laddar ner training dataset
        self.dataset = datasets.CIFAR10(
            root="../data", # Vart datasetet sparas
            train=train, # training data
            download=True, # gör så att det laddas ner automatiskt
            transform=self.transform
        )

    def __len__(self): # Returnerar antal samlples
        return len(self.dataset)
    
    def __getitem__(self, idx): # Returnerar ett sample
        image, label = self.dataset[idx]
        return image, label


dataset = MyDataset()
print(dataset.__len__())
print(dataset.__getitem__(0))