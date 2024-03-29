# Projet implémentation d'un CNN - LeNet-5 sur GPU

### Ce dépôt est un projet scolaire et a été réalisé avec Ossama durant la 3e année à l'ENSEA en spécialité SIA. https://github.com/OssamaChrifi

#### Les objectifs visés sont les suivants :
* Apprendre à utiliser CUDA
* Etudier la complexité des algorithmes et l'accélération obtenue sur GPU par rapport à une exécution sur CPU
* Observer les limites de l'utilisation d'un GPU
* Implémenter "from scratch" un CNN : seulement la partie inférence et non l'entraînement
* Exporter des données depuis un notebook python et les réimporter dans un projet cuda

### Le projet est divisé en 3 parties :
* Partie 1 - Prise en main de Cuda : Multiplication de matrices
* Partie 2 - Premières couches du réseau de neurones LeNet-5 : Convolution 2D, subsampling et fonction d'activation
* Partie 3 - Un peu de Python : Importation du dataset MNIST et affichage des données en console, export des poids

## Installation et mise en place
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
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
L'objectif final est d'implémenter l'inférence d'un CNN très classique : LeNet-5. 
Les informations nécessaires pour comprendre ce réseau de neurone sont apportées par l'article suivant:
https://www.datasciencecentral.com/profiles/blogs/lenet-5-a-classic-cnn-architecture
![Alt text](readme_files/LeNet-5.png)
### Layer 3
Attention, contrairement à ce qui est décrit dans l'article, la 3e couche du CNN prend en compte tous les features pour chaque sortie.

## Contenu
* aide_programmation : dossier contenant des exemples de programmes utiles et fonctionnels
* lenet_5_model : dossier contenant le modèle LeNet-5 à importer dans le notebook Jupyter
* readme_files : fichiers pour les sources de ce readme
* FashionMNIST_weights.h5 : fichier des poids exportés au format h5 (à ignorer, layer_x.bin est utilisé à la place)
* LeNet5.ipynb : notebook jupyter permettant l'entraînement du modèle LeNet-5
* Readme.md : ce readme
* TPx.cu : ficher du code CUDA de la partie 'x'
* TPx : fichier éxecutable de la partie 'x'
* fashionmnist_model.json : fichier des poids exportés au format JSON (à ignorer, layer_x.bin est utilisé à la place)
* layer_x.bin : fichier des poids exportés en brut de la couche 'x' correspondante (avec 'x' = {0,1,2,3,4,5,6,7}) (voir Notes - Partie 3 pour plus d'info)

* train-images.idx3-ubyte : base de données MNIST pour le modèle (chiffres manuscrits)

## Notes - Partie 1
Pour faire la multiplication de matrice de taille nxn, le CPU et le GPU ont o(n^3) opérations à faire.
Le CPU prend beaucoup plus de temps que le GPU à faire les operations.
Le temps d'operation du GPU depend des nombres de block et de threat par block. Plus le nombre de block est grand (threads par block est faible)
plus le temps de calcul sera court car chaque noyau feront des opération en meme temps (calculs parallélisées). Donc la charge de calcul est divisée.
Par exemple, pour deux matrices 1000x1000 le temps serait de 30µs sans parallélisation, tandis qu'avec le GPU (pour 1000 blocks et 1000 threads) le temps est de 1ns environ, ce qui donne un rapport de quasi 30000.
Mais le nombre de blocks n'est pas illimité. Donc avec un nombre grand comme une matrice de taille 2000x2000, le GPU prendra beaucoup plus de temps. Tous les blocks étant occupés, il n'y a nul autre choix que d'allouer plus sur les threads.
Le temps de calcul du GPU pour une matrice très grande n'est donc pas linéaire.

## Notes - Partie 2
Test de la convolution : Prendre un kernel 5x5 rempli de 1 et vérifier que chaque case de la sortie est bien la somme des 25 cases (au même endroit) de l'entrée en les affichant.

Test de la fonction de sous échantillonnage (max pooling) : vérifier la taille de sortie de la matrice par rapport à l'entrée en les affichant.

## Notes - Partie 3
L'entraînement du réseau LeNet-5 n'étant pas réalisé sous cuda mais à travers un script Jupyter (en se servant de Tensorflow), il faut pouvoir exporter les poids et les exploiter sous CUDA. Nous avons opté de les exporter simplemment dans un fichier binaire pour chaque couche layer_x.bin


Exemple pour comprendre son contenu :

Si les poids de la couche 1 sont comme suit : w1[0] = -1.02, w1[1] = -0.81, w1[2] = 2.51 ...
Le fichier layer_x.bin correspondant sera rempli comme suit :
```
101111111000001010001111010111001011111000111000010100011110110001000000001000001010001111010111...
```
En supposant que la lecture du .bin se fait float par float (32 bits), on retrouve bien :
```
10111111100000101000111101011100 | 10111110001110000101000111101100 | 01000000001000001010001111010111...
```
C'est-à-dire :
```
-1.02 -0.81 2.51...
```
