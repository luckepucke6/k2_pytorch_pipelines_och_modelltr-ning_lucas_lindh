import torch

def evaluate(model, dataloader, device):
    """
    Utvärderar modellen och räknar accuracy.
    """

    model.eval() # Sätter modellen i evaluation mode

    correct = 0
    total = 0

    # Stänger av gradient-beräkning
    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Välj klass med högst score
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy