# 🐶 Identification des races de chiens utilisant l'apprentissage par transfert et TensorFlow

Ce projet utilise l'apprentissage automatique pour identifier différentes races de chiens. Il repose sur l'apprentissage par transfert avec un modèle pré-entraîné de TensorFlow Hub et utilise un jeu de données provenant de la compétition Kaggle sur l'identification des races de chiens.

## Flux de travail du projet

Le projet suit un flux de travail standard TensorFlow/Deep Learning :

1. **Acquisition et préparation des données :**
   - Télécharge le jeu de données depuis Kaggle et le stocke localement.
   - Prépare les données en les divisant en ensembles d'entraînement, de validation et de test.
   - Convertit les images en tenseurs numériques pour l'entrée du modèle.

2. **Sélection et entraînement du modèle :**
   - Utilise un modèle pré-entraîné de TensorFlow Hub (`mobilenet_v2_130_224`) pour l'apprentissage par transfert.
   - Construit un modèle Keras avec une couche d'entrée basée sur le modèle pré-entraîné et une couche de sortie pour la classification.
   - Compile le modèle avec la fonction de perte, l'optimiseur et les métriques d'évaluation appropriées.
   - Entraîne le modèle sur les données d'entraînement, en utilisant des rappels pour la visualisation avec TensorBoard et l'arrêt précoce pour éviter le surapprentissage.

3. **Évaluation et amélioration du modèle :**
   - Évalue la performance du modèle sur l'ensemble de validation.
   - Fait des prédictions et les compare avec les étiquettes de vérité terrain.
   - Visualise les prédictions et les probabilités de confiance.
   - Expérimente avec des hyperparamètres et des techniques pour améliorer la précision.

4. **Sauvegarde et partage du modèle :**
   - Sauvegarde le modèle entraîné pour une utilisation future au format `.h5` ou TensorFlow SavedModel.
   - Fournit des instructions pour charger le modèle sauvegardé afin de faire des prédictions ou de le réajuster.

## Caractéristiques principales

- **Apprentissage par transfert :** Accélère l'entraînement en utilisant un modèle pré-entraîné.
- **Support local ou cloud :** Le projet peut être exécuté sur n'importe quel environnement local (par exemple, VS Code, Jupyter Notebook) ou sur des plateformes cloud (par exemple, Google Colab).
- **TensorBoard :** Visualise la performance du modèle et la progression de l'entraînement.
- **Arrêt précoce :** Prévient le surapprentissage et réduit le temps d'entraînement inutile.
- **Visualisation des données :** Inclut des fonctions pour visualiser les données d'entrée, les prédictions et les niveaux de confiance.

## Configuration et utilisation

### Prérequis
Assurez-vous d'avoir les bibliothèques suivantes installées :
- `tensorflow`
- `tensorflow_hub`
- `pandas`
- `scikit-learn`
- `matplotlib`

Vous pouvez les installer avec :
```bash
pip install tensorflow tensorflow-hub pandas scikit-learn matplotlib tensorboard