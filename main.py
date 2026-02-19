import torch
from torch.utils.data import DataLoader
from src.dataset import MyDataset
from src.model import MyModel
from src.train import train

def main():
    # test_dataset = MyDataset(train=False)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # for images, labels in train_loader:
    #     print("Images shape:", images.shape)
    #     print("Labels shape:", labels.shape)
    #     break
    
    # x = torch.randn(32, 3, 32, 32)

    # output = model(x)

    # print("Output shape: ", output.shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Datasetet
    train_dataset = MyDataset(train=True)

    # Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    # Model
    model = MyModel().to(device)

    # Train
    train(
        model=model,
        dataloader=train_loader,
        device=device,
        epochs=10,
        lr=0.001
    )

if __name__ == "__main__":
    main()