import matplotlib.pyplot as plt
from utils import get_dataloaders, get_dataloaders_OOD, ResNet18_CIFAR, get_device
import numpy as np
import torch
from sklearn.metrics import accuracy_score


def analyze_NC1(H, y, classes):

    class_means = []
    for c in classes:
        class_means.append(np.mean(H[y == c], axis=0))

    global_sigma_per_dim = np.mean((H - np.mean(H, axis=0)) ** 2, axis=0)
    global_sigma = float(np.mean(global_sigma_per_dim))

    class_sigmas = []
    for c in classes:
        
        class_sigma_per_dim = np.mean((H[y==c] - np.mean(H[y==c], axis=0)) ** 2, axis=0)
        class_sigmas.append(float(np.mean(class_sigma_per_dim)))

    print("Variance inter-classe (moyenne sur les dimensions) : ")
    print(f"sigma = {global_sigma}")

    print("Variance par intra-classe (moyenne sur les dimensions) : ")
    for i in range(len(classes)):
        print(f"classe {classes[i]} : sigma = {class_sigmas[i]}")


def analyze_NC2(H, y,classes) : 

    H_center = H - np.mean(H, axis = 0)
    class_means = []
    for c in classes:
        class_means.append(np.mean(H_center[y == c], axis=0))

    C = len(classes)
    norms = np.linalg.norm(class_means, axis=1)
    angles = [[ (class_means[i] @ class_means[j])/(norms[i]*norms[j]) for i in range(C)] for j in range(C)]

    print("Distance des centres de classe à l'origine : ")
    for i in range(C) :
        print(f"classe {classes[i]} : Norme = {np.linalg.norm(class_means[i])}")

    print(f"Angle entre les paires de centre (valeur attendue : -1/(C-1) = {1/(1-C):.2f})")
    for i in range(C) :
        print([f"{x:.2f}" for x in angles[i]])
    

def analyze_NC3(H, y, W, classes) : 

    C = len(classes)

    H_center = H - np.mean(H, axis = 0)
    class_means = []
    for c in classes:
        class_means.append(np.mean(H_center[y == c], axis=0))

    class_means = np.array(class_means)
    _class_norm = np.linalg.norm(class_means, axis = 1)
    class_means = class_means/_class_norm[:,None]
    _W_norm = np.linalg.norm(W, axis = 1)
    W = W/(_W_norm[:,None] + 1e-10)


    print("Alignement Poids vs Moyenne (doit tendre vers 1) : ")
    cosines = []
    for i in range(C) : 
        cosines.append(class_means[i] @ W[classes[i]])
        print(f"Classe {classes[i]} : cos = {cosines[i]:.2f}")

    

def analyze_NC4(H, y, logits, classes) : 
    C = len(classes)

    pred_idx = np.argmax(logits, axis = 1)

    nearest_tabs = []
    for c in classes:
        mean = np.mean(H[y == c], axis=0)
        dist_class = np.linalg.norm(H-mean, axis = 1)
        nearest_tabs.append(dist_class)

    nearest_class = np.array(nearest_tabs)
    nearest_idx = np.argmin(nearest_class, axis = 0)

    preds_model = np.array([classes[i] for i in pred_idx])
    preds_nearest = np.array([classes[i] for i in nearest_idx])
    print(f"Accuracy du modèle : {100*accuracy_score(y, preds_model):.2f}")
    print(f"Accuracy pour la distance euclidienne : {100* accuracy_score(y, preds_nearest):.2f}")
    print(f"Taux d'accord : {100* np.mean(preds_model == preds_nearest):.2f}")


      

def analyze_NC5(H, y, classes, H_ood):  
    global_mean = np.mean(H, axis=0)

    class_means = []
    for c in classes:
        class_means.append(np.mean(H[y == c], axis=0))
    M = np.array(class_means)
    
    M_centered = M - global_mean
    H_id_centered = H - global_mean
    H_ood_centered = H_ood - global_mean

    eps = 1e-10
    M_norm = M_centered / (np.linalg.norm(M_centered, axis=1, keepdims=True) + eps)
    H_id_norm = H_id_centered / (np.linalg.norm(H_id_centered, axis=1, keepdims=True) + eps)
    H_ood_norm = H_ood_centered / (np.linalg.norm(H_ood_centered, axis=1, keepdims=True) + eps)

    cos_id_M = np.abs(H_id_norm @ M_norm.T) 
    cos_ood_M = np.abs(H_ood_norm @ M_norm.T)

    score_id = np.mean(np.max(cos_id_M, axis=1))
    score_ood = np.mean(np.max(cos_ood_M, axis=1))

    print(f"Alignement moyen ID (centré)  : {score_id:.4f}")
    print(f"Alignement moyen OOD (centré) : {score_ood:.4f}")
    
    return score_id, score_ood



def analyze_collapse() : 
    device = get_device()
    train_loader, test_loader = get_dataloaders()
    model = ResNet18_CIFAR().load_from_path("resnet18_cifar.pth").to(device)
    model.eval()

    features_list = []
    labels_list = []
    logits_list = []
    
    with torch.no_grad():
        for imgs, lbls in train_loader : 
            imgs = imgs.to(device)
            logits, feats = model(imgs)
            features_list.append(feats.cpu().numpy())
            labels_list.append(lbls.cpu().numpy())
            logits_list.append(logits.cpu().numpy())

    
    H = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    logits = np.concatenate(logits_list, axis=0)

    W = model.fc.weight.detach().cpu().numpy()
    b = model.fc.bias.detach().cpu().numpy()

    classes = np.unique(y)

    print("////// NC1 //////")
    analyze_NC1(H, y, classes)
    print("////// NC2 //////")
    analyze_NC2(H, y, classes)
    print("////// NC3 //////")
    analyze_NC3(H, y, W, classes)
    print("////// NC4 //////")
    analyze_NC4(H, y, logits, classes)
    print("////// NC5 //////")
    ood_loader = get_dataloaders_OOD(batch_size=128)
    features_ood_list = []
    with torch.no_grad():
        for imgs, _ in ood_loader:
            imgs = imgs.to(device)
            _, feats = model(imgs)
            features_ood_list.append(feats.cpu().numpy())
    H_ood = np.concatenate(features_ood_list, axis=0)
    analyze_NC5(H, y, classes, H_ood)


if __name__ == "__main__" : 
    analyze_collapse()


