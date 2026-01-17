#!/bin/bash

echo "=== [BOOTSTRAP EMR] Installation TensorFlow + tree ==="

# Met à jour pip pour éviter des erreurs d'installation
sudo /usr/bin/python3 -m pip install --upgrade pip

# Installe TensorFlow et tree dans l'environnement système
sudo /usr/bin/python3 -m pip install tensorflow==2.16.1 tree --no-cache-dir

# Ajoute les chemins Python système au PYTHONPATH global
echo 'export PYTHONPATH=/usr/local/lib/python3.9/site-packages:/usr/local/lib64/python3.9/site-packages:$PYTHONPATH' | sudo tee -a /etc/environment
echo 'export PYTHONPATH=/usr/local/lib/python3.9/site-packages:/usr/local/lib64/python3.9/site-packages:$PYTHONPATH' | sudo tee -a /etc/profile.d/pythonpath.sh

# Recharge les variables d'environnement (sera appliqué au prochain shell, mais Livy a besoin d'un redémarrage manuel)
source /etc/environment

# Redémarre Livy pour qu'il hérite de la nouvelle variable d'environnement
sudo systemctl restart livy-server

# Vérification des imports
echo "[TEST] Import TensorFlow"
sudo /usr/bin/python3 -c "import tensorflow as tf; print('TensorFlow OK:', tf.__version__)"
echo "[TEST] Import keras.model"
sudo /usr/bin/python3 -c "from tensorflow.keras.models import Sequential; print('Keras OK')"
echo "[TEST] Import tree"
sudo /usr/bin/python3 -c "import tree; print('Tree OK')"

echo "=== [BOOTSTRAP EMR] Fin installation TensorFlow ==="
