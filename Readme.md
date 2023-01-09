# Projet implémentation d'un CNN - LeNet-5 sur GPU

### Ce dépôt est projet scolaire et a été réalisé avec Ossama durant la 3e année à l'ENSEA. https://github.com/OssamaChrifi

#### Les objectifs visés sont les suivants :
* Apprendre à utiliser CUDA
* Etudier la complexité de vos algorithmes et l'accélération obtenue sur GPU par rapport à une éxécution sur CPU
* Observer les limites de l'utilisation d'un GPU
* Implémenter "from scratch" un CNN : seulement la partie inférence et non l'entrainement
* Exporter des données depuis un notebook python et les réimporter dans un projet cuda

### Le projet est divisé en 3 parties :
* Partie 1 - Prise en main de Cuda : Multiplication de matrices
* Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D, subsampling et fonction d'activation
* Partie 3 - Un peu de Python : Importation du dataset MNIST et affichage des données en console, export des poids

## Installation et mise en place
### Installation
```sh
git clone https://github.com/StephaneYang/CNN_LeNet-5_sur_GPU
```
### Compilation
```sh
nvcc TPx.cu -o TPx
# avec 'x' = {1,2,3} selon la partie désirée
```
### Exécution
```sh
./TPx
# avec 'x' = {1,2,3} selon la partie désirée
```

## Implémentation d'un CNN
L'objectif final est d'implémenter l'inférence d'un CNN très classique : LeNet-5
La lecture de l'article suivant apporte les informations nécessaires pour comprendre ce réseau de neurone :
https://www.datasciencecentral.com/profiles/blogs/lenet-5-a-classic-cnn-architecture
![Alt text](readme_files/LeNet-5.png)
### Layer 3
Attention, contrairement à ce qui est décrit dans l'article, la 3e couche du CNN prend en compte tous les features pour chaque sortie.

## Contenu
* aide_programmation : dossier contenant des exemmples de programmes utiles et fonctionnels
* lenet_5_model : dossier contenant le modèle LeNet-5 à importer dans le notebook Jupyter
* readme_files : fichiers pour les sources de ce readme
* FashionMNIST_weights.h5 : fichier des poids exportés au format h5(à ignorer)
* LeNet5.ipynb : notebook jupyter permettant l'entraînement du modèle LeNet-5
* Readme.md : ce readme
* TPx.cu : ficher du code CUDA de la partie 'x'
* TPx : fichier éxecutable de la partie 'x'
* layer_x.bin : fichier des poids exportés en brut de la couche 'x' correspondante (avec 'x' = {0,1,2,3,4,5,6,7})
Si les poids de la couche 1 sont comme suit : w1[0] = -1.02, w1[1] = -0.81, w1[2] = 2.51 ...
Le fichier .bin correspondant sera comme suit :
```
101111111000001010001111010111001011111000111000010100011110110001000000001000001010001111010111...
```
En supposant que la lecture du .bin se fait float par float (32 bits), on retrouve bien :
```
-1.02 -0.81 2.51...
```
* train-images.idx3-ubyte : base de données MNIST pour le modèle (chiffres manuscrits)

## Notes
Pour faire la multiplication de matrice de taille n x n, le CPU et le GPU à n^3 operation à faire.
Le CPU prend beaucoup plus de temps que le GPU à faire les operations.
Le temps d'operation du GPU depend des nombres de block et de threat par block. Plus le nombre de block est threat par block est grand
plus le temps de calcul sera grand car chaque noyau font des opération en meme temps. Donc la charge de calcule est divisé.
Mais le nombre de block n'est pas limité donc avec un nombre grand comme une matrice de taille 2000 x 2000, le GPU prendra beaucoup plus de temps.
Le temps de calcul du GPU pour une matrice grande n'est pas linéaire.
