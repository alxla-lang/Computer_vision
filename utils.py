import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes = 100) :
        super(ResNet18_CIFAR, self).__init__()

        self.backbone = resnet18(weights=None)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x) : 
        features = self.backbone(x)
        logits = self.fc(features)
        return logits, features

    def load_from_path(self, path) : 
        self.load_state_dict(torch.load(path))
        return self

    def save_to_path(self, path) : 
        torch.save(self.state_dict(), path)

    
def get_dataloaders(batch_size = 128) : 
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader


def get_dataloaders_OOD(batch_size=128, num_workers=4):
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    ood_dataset = torchvision.datasets.SVHN(
        root="./data", split="test", download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        ood_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
