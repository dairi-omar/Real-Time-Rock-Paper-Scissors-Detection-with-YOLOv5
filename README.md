#  Real-Time Rock-Paper-Scissors Detection with YOLOv5

Ce projet implémente un système de vision par ordinateur capable de détecter et de classifier les gestes du jeu **Pierre-Papier-Ciseaux** en temps réel via une webcam. Le modèle est basé sur l'architecture **YOLOv5** et entraîné sur un dataset personnalisé.

## Fonctionnalités

* **Pipeline complet :** Préparation des données, conversion des annotations (CSV vers YOLO format) et entraînement.
* **Modèle performant :** Utilisation du Transfer Learning sur le modèle `yolov5su` pour une convergence rapide.
* **Inférence Live :** Script Python pour la détection en temps réel via webcam utilisant OpenCV.
* **Visualisation :** Génération de graphiques de performance (Matrice de confusion, Courbes F1, mAP).

## Structure du Projet

* `noteBook.ipynb` : Le carnet Jupyter contenant tout le processus d'entraînement :
    * Chargement et analyse du dataset.
    * Conversion des labels au format normalisé YOLO.
    * Entraînement du modèle (Training).
    * Évaluation des performances.
* `test.py` : Script d'application finale. Il charge le modèle entraîné (`best.pt`) et lance la détection sur le flux vidéo de la caméra.
