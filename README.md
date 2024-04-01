# Style Transfer Application - ARTIFY

## Objectif : Revisiter le concept d'art à travers l'intelligence artificielle
![1_8bbp3loQjkLXaIm_QBfD8w](https://github.com/Joseph-Willson/Projet_Linux_Image/assets/102141518/6ef79402-0f01-4e07-9562-96268c58796b)


Dans un monde où la technologie façonne nos interactions quotidiennes, notre projet se distingue par sa capacité à fusionner l'intelligence artificielle avec l'expression artistique. À l'intersection de la science des données et de la créativité humaine, nous avons conçu un système novateur de transfert d'images qui transcende les limites traditionnelles de la manipulation visuelle. En explorant les subtilités de l'apprentissage automatique, notre projet permet non seulement de transférer des éléments d'une image à une autre, mais également de capturer l'essence même de différents styles artistiques et de les intégrer de manière fluide. C'est une invitation à plonger dans un univers où les frontières entre l'homme et la machine s'estompent, où la technologie devient un outil d'expression aussi bien artistique que fonctionnel. Notre initiative ouvre ainsi de nouveaux horizons pour la création visuelle, offrant aux artistes et aux professionnels de multiples domaines un terrain fertile pour explorer, expérimenter et innover. Préparez-vous à être émerveillé par la symbiose entre la science et l'art, où chaque pixel devient une toile sur laquelle se dessine l'avenir de l'imagerie numérique.



### Afin d'exécuter le projet, nous vous invitons à suivre les étapes suivantes qui vous permettront d'accéder à notre projet

#### Commande à exécuter pour lancer le projet :

#### cloner le projet

`git clone https://github.com/Joseph-Willson/Projet_Linux_Image.git                               `

#### Entrez dans le projet

`cd Projet_Linux_Image                                            `

#### Exécuter la commande bash ci-dessous pour installer les bibliothèques, dépendances et environnement virtuel

`bash install.sh                                                  `


#### Puis exécuter la commande bash ci-dessous pour lancer l'application.

`bash launch.sh                                                   `

  
Une fois arrivé sur la page Web, vous pourrez selectionner les images que vous souhaitez mixer et définir le niveau de qualité (Faible, Moyen, Elevé)  
Attention : le niveau elevé risque de prendre tu temps à tourner 😁

  
### Dans le cadre de notre projet, nous avons créé un Dockerfile pour la gestion des dépendances, permettant de lancer le projet dans l'environnement dans lequel il a été travaillé.

#### Vous pourrez suivre les étapes suivantes pour lancer le projet à partir du Dockerfile :

#### cloner le projet

`git clone https://github.com/Joseph-Willson/Projet_Linux_Image.git                               `

#### Entrez dans le projet

`cd Projet_Linux_Image                                            `

#### Construction du conteneur Docker

`docker build -t nomimage:image .                                 `

#### lancement du conteneur

`docker run -p 8084:8084 nomimage:image                           `


## Enjoy 😊

![1699444010913 2~tplv-6rr7idwo9f-image (1)](https://github.com/Joseph-Willson/Projet_Linux_Image/assets/102141518/f68ddf02-3ce0-43c3-a046-f0262a5bd599)




