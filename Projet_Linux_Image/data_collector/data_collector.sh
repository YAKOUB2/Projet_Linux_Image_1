#!/bin/bash

# Activer l'environnement virtuel
source ./venv/bin/activate

# Récupérer l'image avec curl et la placer dans un dossier source
curl -o Elia_Rahari.jpeg "https://media.licdn.com/dms/image/D4E03AQH3np9QMfiZUw/profile-displayphoto-shrink_800_800/0/1697620991340?e=1713398400&v=beta&t=xZ8rIzhWsBeZTjDiRNAWzQrqycFATm--m7gxrcCG1hw"
curl -o Cristiano_Ronaldo.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Cristiano_Ronaldo_playing_for_Al_Nassr_FC_against_Persepolis%2C_September_2023_%28cropped%29.jpg/375px-Cristiano_Ronaldo_playing_for_Al_Nassr_FC_against_Persepolis%2C_September_2023_%28cropped%29.jpg"


curl -o Starry_night.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/800px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
curl -o Composition.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/800px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg"
curl -o Mona_lisa.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_natural_color.jpg/1200px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_natural_color.jpg"
curl -o Wave_Kangawa.jpg "https://upload.wikimedia.org/wikipedia/commons/a/a5/Tsunami_by_hokusai_19th_century.jpg"
curl -o Scream.jpg "https://upload.wikimedia.org/wikipedia/commons/3/34/Edvard-Munch-The-Scream.jpg"
curl -o Water_lillies.jpg "https://upload.wikimedia.org/wikipedia/commons/a/aa/Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg"

# Créer le dossier source s'il n'existe pas
mkdir -p content_images
mkdir -p style_images
mkdir -p exemple

# Déplacer l'image téléchargée dans le dossier source
mv Elia_Rahari.jpeg content_images/
mv Cristiano_Ronaldo.jpg content_images/


mv Starry_night.jpg style_images/
mv Composition.jpg style_images/
mv Mona_lisa.jpg style_images/
mv Wave_Kangawa.jpg style_images/
mv Scream.jpg style_images/
mv Water_lillies.jpg style_images/

curl -o Elia_Rahari.jpeg "https://media.licdn.com/dms/image/D4E03AQH3np9QMfiZUw/profile-displayphoto-shrink_800_800/0/1697620991340?e=1713398400&v=beta&t=xZ8rIzhWsBeZTjDiRNAWzQrqycFATm--m7gxrcCG1hw"
curl -o Starry_night.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/800px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"

mv Elia_Rahari.jpeg exemple/
mv Starry_night.jpg exemple/

# Donner des permissions au dossier source
chmod +x ./data_collector/data_collector.sh
