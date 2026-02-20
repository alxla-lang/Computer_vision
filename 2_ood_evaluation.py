import torch
import torch.nn as nn
from torchvision import transforms  
from torchvision import datasets
import numpy as np
from utils import get_device, get_dataloaders, ResNet18_CIFAR
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EmpiricalCovariance


def get_OOD_loader(batch_size = 100) : 
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    ood_set = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    loader = torch.utils.data.DataLoader(ood_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return loader


def extract_features_and_logits(model, loader, device) : 
    model.eval()
    all_features = []
    all_logits = []
    with torch.no_grad() : 
        for inputs, _ in loader : 
            inputs = inputs.to(device)
            logits, features = model(inputs)
            all_logits.append(logits)
            all_features.append(features)
    return torch.cat(all_features), torch.cat(all_logits)

def score_msp(logits) : 
    probs = torch.softmax(logits, dim=-1)
    return torch.max(probs, dim=-1).values.cpu().numpy()

def score_mls(logits) : 
    return torch.max(logits, dim=-1).values.cpu().numpy()

def score_energy(logits) : 
    return -torch.logsumexp(logits, dim=-1).cpu().numpy()

def setup_mahalanobis(train_features, train_labels):
    """ Calcule les moyennes de classe et la matrice de précision (inverse covariance) """
    train_features = train_features.cpu().numpy()
    train_labels = train_labels.cpu().numpy()
    classes = np.unique(train_labels)
    
    class_means = {}
    centered_features = []
    
    for c in classes:
        mask = (train_labels == c)
        features_c = train_features[mask]
        mean_c = np.mean(features_c, axis=0)
        class_means[c] = mean_c
        centered_features.append(features_c - mean_c)
        
    centered_features = np.concatenate(centered_features)
    cov = EmpiricalCovariance(assume_centered=True).fit(centered_features)
    precision = cov.precision_
    
    return class_means, precision

def score_mahalanobis(features, class_means, precision_matrix):

    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    
    original_shape = features.shape
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    elif len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    
    means_array = np.array([mean for mean in class_means.values()])
    
    if features.shape[1] != means_array.shape[1]:
        raise ValueError(
            f"Incompatible feature dimensions!\n"
            f"  features.shape (after reshape): {features.shape} (original: {original_shape})\n"
            f"  means_array.shape: {means_array.shape}\n"
            f"  Expected feature_dim: {means_array.shape[1]}, got: {features.shape[1]}"
        )
    
    scores = []
    for i in range(len(features)):
        x = features[i]
        if len(x.shape) > 1:
            x = x.flatten()
        diff = x - means_array
        dists = np.sum((diff @ precision_matrix) * diff, axis=1)
        scores.append(-np.min(dists))
        
    return np.array(scores)


def main():
    device = get_device()
    print(f"Évaluation sur : {device}")

    train_loader, test_loader = get_dataloaders()
    ood_loader = get_OOD_loader()
    
    model = ResNet18_CIFAR()
    model.load_state_dict(torch.load("resnet18_cifar.pth", map_location=device))
    model = model.to(device)
    model.eval()

    print("Extraction des features du TRAIN set (pour Mahalanobis)...")
    train_feats = []
    train_lbls = []
    with torch.no_grad():
        for x, y in train_loader:
            _, f = model(x.to(device))
            train_feats.append(f.cpu())
            train_lbls.append(y)
    train_feats = torch.cat(train_feats)
    train_lbls = torch.cat(train_lbls)
    
    print("Calcul des statistiques (Moyennes & Covariance)...")
    class_means, precision = setup_mahalanobis(train_feats, train_lbls)

    print("Extraction ID (Test CIFAR-10)...")
    id_features, id_logits = extract_features_and_logits(model, test_loader, device)
    
    print("Extraction OOD (SVHN)...")
    # On limite la taille OOD à la taille ID pour équilibrer le calcul AUROC
    ood_features, ood_logits = extract_features_and_logits(model, ood_loader, device)
    ood_features = ood_features[:len(id_features)]
    ood_logits = ood_logits[:len(id_logits)]

    # B. Calcul des Scores et AUROC
    # Convention : Label 1 = In-Distribution, Label 0 = OOD
    y_true = np.concatenate([np.ones(len(id_logits)), np.zeros(len(ood_logits))])
    
    methods = {
        "MSP": score_msp,
        "MLS": score_mls,
        "Energy": score_energy
    }

    print("\n--- RÉSULTATS (AUROC) ---")
    print("L'AUROC doit être proche de 1.0 pour une détection parfaite.")
    
    # 1. Méthodes basées sur les Logits
    for name, func in methods.items():
        id_score = func(id_logits)
        ood_score = func(ood_logits)
        scores = np.concatenate([id_score, ood_score])
        auroc = roc_auc_score(y_true, scores)
        print(f"{name:15s} : {auroc:.4f}")

    # 2. Méthode Mahalanobis (basée sur les Features)
    id_mah = score_mahalanobis(id_features, class_means, precision)
    ood_mah = score_mahalanobis(ood_features, class_means, precision)
    scores_mah = np.concatenate([id_mah, ood_mah])
    auroc_mah = roc_auc_score(y_true, scores_mah)
    print(f"{'Mahalanobis':15s} : {auroc_mah:.4f}")

if __name__ == "__main__":
    main()