#!/usr/bin/env bash
#SBATCH --job-name=job_test_thomas         # Nom du job [cite: 120]
#SBATCH --partition=ENSTA-h100        # Utilisation de la partition H100 [cite: 71, 121]
#SBATCH --gres=gpu:1                  # 1 GPU alloué [cite: 124]
#SBATCH --cpus-per-task=4             # 4 CPUs pour le chargement des données [cite: 125]
#SBATCH --time=02:00:00               # Temps limite (HH:MM:SS) [cite: 126, 133, 186]
#SBATCH --output=./output/evaluation_results%j.out     # Journal de sortie [cite: 127, 132]

source ~/miniconda3/etc/profile.d/conda.sh
conda activate python_env
echo "Lancement de l'entraînement sur le cluster..."
python 2_ood_evaluation.py
echo "Job terminé avec succès."