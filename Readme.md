# Projet implémentation d'un CNN - LeNet-5 sur GPU

### Ce dépôt est projet scolaire et a été réalisé avec Ossama durant la 3e année à l'ENSEA. https://github.com/OssamaChrifi

#### Les objectifs visés sont les suivants :
* Apprendre à utiliser CUDA
* Etudier la complexité de vos algorithmes et l'accélération obtenue sur GPU par rapport à une éxécution sur CPU
* Observer les limites de l'utilisation d'un GPU
* Implémenter "from scratch" un CNN : seulement la partie inférence et non l'entrainement
* Exporter des données depuis un notebook python et les réimporter dans un projet cuda

## Implémentation d'un CNN
L'objectif final est d'implémenter l'inférence d'un CNN très classique : LeNet-5
La lecture de l'article suivant apportera les informations nécessaires pour comprendre ce réseau de neurone :
https://www.datasciencecentral.com/profiles/blogs/lenet-5-a-classic-cnn-architecture

## Notes
Pour faire la multiplication de matrice de taille n x n, le CPU et le GPU à n^3 operation à faire.
Le CPU prend beaucoup plus de temps que le GPU à faire les operations.
Le temps d'operation du GPU depend des nombres de block et de threat par block. Plus le nombre de block est threat par block est grand
plus le temps de calcul sera grand car chaque noyau font des opération en meme temps. Donc la charge de calcule est divisé.
Mais le nombre de block n'est pas limité donc avec un nombre grand comme une matrice de taille 2000 x 2000, le GPU prendra beaucoup plus de temps.
Le temps de calcul du GPU pour une matrice grande n'est pas linéaire.
