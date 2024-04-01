#!/bin/bash

virtualenv venv

source ./venv/bin/activate

# Installer les dépendances Python via pip
pip install -r requirements.txt

# Installation de Streamlit
pip install streamlit

# Installation des autres dépendances
pip install tensorflow matplotlib numpy IPython


# Assurez-vous que votre script est exécutable
chmod +x install.sh


