import torch
import torch.nn as nn
import torch.optim as optim

def train(model, dataloader, device, epochs=5, lr=0.001):
    """
    Docstring for train
    
    model: MyModel
    dataloader: training DataLoader (dataloader = delar upp datasetet i mindre bitar och levererar dem en batch i taget till träningen)
    device: cpu eller cuda
    epochs: antal varv genom datasetet
    lr: learning rate
    """

    # Sätter modellen i träningsläge
    for epoch in range(epochs):
        model.train()

    # Loss funktion, beräknar felet
    criterion = nn.CrossEntropyLoss()

    # Optimizer, justerar modellens vikter och bias steg för steg för att minimera loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        running_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 1. Nollställer gradients
            optimizer.zero_grad()

            # 2. Forward pass
            outputs = model(images)

            # 3. Beräkna loss
            loss = criterion(outputs, labels)

            # 4. Backward pass, räknas baklänges från felet och kollar hur varje vikt bidrog. Då kan den justera vikterna så det blir optimerat.
            loss.backward()

            # 5. Uppdaterar vikter efter vi fått resultatet från loss, så den blir lite bättre för nästa omgång
            optimizer.step()

            # .item() -> hämtar ut värdet från en tensor och gör det till ett vanligt tal
            running_loss += loss.item()

        # Beräknar medelfelet för hela epochen. Running loss = totala felet av alla batchers fel, och len(dataloader) är antalet batcher
        avg_loss = running_loss / len(dataloader)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    print("Träning avklarad")