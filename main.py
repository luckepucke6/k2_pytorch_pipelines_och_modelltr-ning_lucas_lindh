import torch
import yaml
from torch.utils.data import DataLoader
from src.dataset import MyDataset
from src.model import MyModel
from src.train import train
from src.evaluate import evaluate

def main():

    # Läser hyperparametrar
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    batch_size = params["train"]["batch_size"]
    epochs = params["train"]["epochs"]
    learning_rate = params["train"]["learning_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Datasetet
    train_dataset = MyDataset(train=True)

    # Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Model
    model = MyModel().to(device)

    # Train
    train(
        model=model,
        dataloader=train_loader,
        device=device,
        epochs=epochs,
        lr=learning_rate
    )

    # Validering
    val_dataset = MyDataset(train=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Evaluate
    accuracy = evaluate(model, val_loader, device)

    print("Validation accuracy:", accuracy)

if __name__ == "__main__":
    main()