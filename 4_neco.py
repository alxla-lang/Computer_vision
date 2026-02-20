import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import get_device, get_dataloaders, get_dataloaders_OOD, ResNet18_CIFAR

def extract_features(model, loader, device):
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, feats = model(x)
            features_list.append(feats.cpu().numpy())
            labels_list.append(y.cpu().numpy())
            
    return np.concatenate(features_list), np.concatenate(labels_list)

def setup_neco(train_features, train_labels):

    global_mean = np.mean(train_features, axis=0)
    
    classes = np.unique(train_labels)
    class_centers = []
    
    for c in classes:
        mu_c = np.mean(train_features[train_labels == c], axis=0)
        mu_c_centered = mu_c - global_mean
        class_centers.append(mu_c_centered)
        
    class_centers = np.array(class_centers)
    
    norms = np.linalg.norm(class_centers, axis=1, keepdims=True)
    normalized_centers = class_centers / (norms + 1e-10)
    
    return global_mean, normalized_centers

def compute_neco_score(features, global_mean, normalized_centers):

    features_centered = features - global_mean
    norms = np.linalg.norm(features_centered, axis=1, keepdims=True)
    features_norm = features_centered / (norms + 1e-10)
    
    cosines = np.dot(features_norm, normalized_centers.T)
    neco_scores = np.max(cosines, axis=1)
    
    return neco_scores


def main():
    device = get_device()
    print(f"--- Évaluation NECO ---")

    train_loader, test_loader = get_dataloaders(batch_size=128)
    ood_loader = get_dataloaders_OOD(batch_size=128)
    
    model = ResNet18_CIFAR().to(device)
    model.load_state_dict(torch.load("resnet18_cifar.pth", map_location=device))

    print("Extraction des features du TRAIN")
    train_feats, train_lbls = extract_features(model, train_loader, device)
    global_mean, normalized_centers = setup_neco(train_feats, train_lbls)

    print("Extraction des features ID")
    id_feats, _ = extract_features(model, test_loader, device)
    
    print("Extraction des features OOD")
    ood_feats, _ = extract_features(model, ood_loader, device)
    ood_feats = ood_feats[:len(id_feats)]

    scores_id = compute_neco_score(id_feats, global_mean, normalized_centers)
    scores_ood = compute_neco_score(ood_feats, global_mean, normalized_centers)

    print(f"Score NECO moyen ID  : {np.mean(scores_id):.4f} (Doit être proche de 1)")
    print(f"Score NECO moyen OOD : {np.mean(scores_ood):.4f} (Doit être plus faible)")

    y_true = np.concatenate([np.ones(len(scores_id)), np.zeros(len(scores_ood))])
    y_scores = np.concatenate([scores_id, scores_ood])
    
    auroc = roc_auc_score(y_true, y_scores)
    
    print("\nRÉSULTAT FINAL")
    print(f"NECO AUROC : {auroc:.4f}")


if __name__ == "__main__":
    main()