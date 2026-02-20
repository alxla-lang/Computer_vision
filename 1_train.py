import torch
import torch.optim as optim
import torch.nn as nn
from utils import get_dataloaders, ResNet18_CIFAR, get_device
import time
import matplotlib.pyplot as plt
def main() : 
    device = get_device()
    print(f"Entraînement sur {device}")

    #Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9
    EPOCHS = 100

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    model = ResNet18_CIFAR().to(device)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Entraînement sur {EPOCHS} époques")
    start_time = time.time()
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()

        scheduler.step()

        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%")
        train_losses.append(running_loss/len(train_loader))
        train_accuracies.append(acc)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                total += labels.size(0)
                correct += outputs.argmax(dim=1).eq(labels).sum().item()
        acc = 100.0 * correct / total
        test_losses.append(running_loss/len(test_loader))
        test_accuracies.append(acc)

    total_time = time.time() - start_time
    print(f"Temps d'entraînement : {total_time:.2f}s")

    model.save_to_path("resnet18_cifar.pth")

    
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")

    plt.legend()
    plt.savefig("train_accuracy_test_accuracy.png")

    plt.clf()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.savefig("train_loss_test_loss.png")



if __name__ == "__main__":
    main()

