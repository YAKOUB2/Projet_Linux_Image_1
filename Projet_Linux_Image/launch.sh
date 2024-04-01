echo "******************* Initialisation de l'application *******************"
source ./venv/bin/activate
echo "******************* l'environnement virtuel est activé *******************"

echo "******************* Collecte des données *******************"
bash ./data_collector/data_collector.sh
echo "******************* Données collectées *******************"

#echo "******************* Lecture et affichage *******************"
#bash ./data_integrator/data_integrator.sh

echo "******************* Affichage de l'application *******************"
python3 -m streamlit run ./Webapp/app.py --server.port 8084




